def classify_hvac_capacity(heating_kw, cooling_kw, building_area_m2, 
                          building_height_m, estimated_apartments, 
                          floor_height=3.0, confidence_threshold=0.7):
    """
    Determine if HVAC capacity data refers to a single apartment or entire building.
    
    Parameters:
    - heating_kw: Heating capacity in kW
    - cooling_kw: Cooling capacity in kW (optional, can be None)
    - building_area_m2: Total building floor area
    - building_height_m: Building height
    - estimated_apartments: Estimated number of apartments
    - floor_height: Typical floor height (default 3.0m)
    - confidence_threshold: Minimum confidence for classification (0-1)
    
    Returns:
    - dict with classification results and confidence scores
    """
    
    # Typical capacity ranges for Italian apartments (kW)
    APARTMENT_RANGES = {
        'heating': {'min': 2.5, 'typical_min': 4, 'typical_max': 18, 'max': 25},
        'cooling': {'min': 1.5, 'typical_min': 3, 'typical_max': 15, 'max': 20}
    }
    
    # Calculate building metrics
    estimated_floors = max(1, round(building_height_m / floor_height))
    avg_apartment_area = building_area_m2 / max(1, estimated_apartments)
    
    # Expected capacity per apartment based on area (simplified heuristic)
    expected_heating_per_apt = max(3, min(25, 0.08 * avg_apartment_area + 2))
    expected_cooling_per_apt = max(2, min(18, 0.06 * avg_apartment_area + 1.5))
    
    # Calculate expected total building capacity
    expected_building_heating = expected_heating_per_apt * estimated_apartments
    expected_building_cooling = expected_cooling_per_apt * estimated_apartments
    
    results = {
        'classification': None,
        'confidence': 0.0,
        'reasoning': [],
        'metrics': {
            'avg_apartment_area_m2': round(avg_apartment_area, 1),
            'estimated_floors': estimated_floors,
            'expected_heating_per_apt': round(expected_heating_per_apt, 1),
            'expected_cooling_per_apt': round(expected_cooling_per_apt, 1),
            'expected_building_heating': round(expected_building_heating, 1),
            'expected_building_cooling': round(expected_building_cooling, 1)
        }
    }
    
    # Analyze heating capacity
    heating_scores = {'apartment': 0, 'building': 0}
    heating_reasoning = []
    
    # Check if heating falls in apartment range
    if (APARTMENT_RANGES['heating']['min'] <= heating_kw <= 
        APARTMENT_RANGES['heating']['max']):
        heating_scores['apartment'] += 0.6
        heating_reasoning.append(f"Heating {heating_kw}kW within apartment range")
        
        if (APARTMENT_RANGES['heating']['typical_min'] <= heating_kw <= 
            APARTMENT_RANGES['heating']['typical_max']):
            heating_scores['apartment'] += 0.3
            heating_reasoning.append("Within typical apartment heating range")
    
    # Check if heating is too high for single apartment
    if heating_kw > APARTMENT_RANGES['heating']['max']:
        heating_scores['building'] += 0.8
        heating_reasoning.append(f"Heating {heating_kw}kW too high for single apartment")
    
    # Compare with expected building capacity
    building_heating_ratio = heating_kw / expected_building_heating
    if 0.5 <= building_heating_ratio <= 2.0:
        heating_scores['building'] += 0.4
        heating_reasoning.append(f"Heating matches expected building capacity (ratio: {building_heating_ratio:.2f})")
    
    # Compare with per-apartment expectation
    apt_heating_ratio = heating_kw / expected_heating_per_apt
    if 0.7 <= apt_heating_ratio <= 1.5:
        heating_scores['apartment'] += 0.4
        heating_reasoning.append(f"Heating matches expected apartment capacity (ratio: {apt_heating_ratio:.2f})")
    
    # Analyze cooling capacity if available
    cooling_scores = {'apartment': 0, 'building': 0}
    cooling_reasoning = []
    
    if cooling_kw is not None:
        # Check cooling ranges
        if (APARTMENT_RANGES['cooling']['min'] <= cooling_kw <= 
            APARTMENT_RANGES['cooling']['max']):
            cooling_scores['apartment'] += 0.6
            cooling_reasoning.append(f"Cooling {cooling_kw}kW within apartment range")
            
            if (APARTMENT_RANGES['cooling']['typical_min'] <= cooling_kw <= 
                APARTMENT_RANGES['cooling']['typical_max']):
                cooling_scores['apartment'] += 0.3
                cooling_reasoning.append("Within typical apartment cooling range")
        
        if cooling_kw > APARTMENT_RANGES['cooling']['max']:
            cooling_scores['building'] += 0.8
            cooling_reasoning.append(f"Cooling {cooling_kw}kW too high for single apartment")
        
        # Compare with building expectation
        building_cooling_ratio = cooling_kw / expected_building_cooling
        if 0.5 <= building_cooling_ratio <= 2.0:
            cooling_scores['building'] += 0.4
            cooling_reasoning.append(f"Cooling matches expected building capacity (ratio: {building_cooling_ratio:.2f})")
        
        # Compare with apartment expectation
        apt_cooling_ratio = cooling_kw / expected_cooling_per_apt
        if 0.7 <= apt_cooling_ratio <= 1.5:
            cooling_scores['apartment'] += 0.4
            cooling_reasoning.append(f"Cooling matches expected apartment capacity (ratio: {apt_cooling_ratio:.2f})")
    
    # Additional heuristics
    additional_reasoning = []
    
    # Very small buildings are more likely to be single apartments
    if estimated_apartments <= 2:
        heating_scores['apartment'] += 0.2
        cooling_scores['apartment'] += 0.2
        additional_reasoning.append("Small building favors apartment-level data")
    
    # Very large buildings with small per-apartment capacity suggest building-level
    if estimated_apartments >= 10 and heating_kw / estimated_apartments < 3:
        heating_scores['building'] += 0.3
        cooling_scores['building'] += 0.3
        additional_reasoning.append("Large building with low per-unit capacity suggests building-level data")
    
    # Combine scores (weight heating more heavily as it's more reliable)
    heating_weight = 0.7 if cooling_kw is not None else 1.0
    cooling_weight = 0.3 if cooling_kw is not None else 0.0
    
    final_apartment_score = (heating_scores['apartment'] * heating_weight + 
                           cooling_scores['apartment'] * cooling_weight)
    final_building_score = (heating_scores['building'] * heating_weight + 
                          cooling_scores['building'] * cooling_weight)
    
    # Determine classification
    if final_apartment_score > final_building_score:
        if final_apartment_score >= confidence_threshold:
            results['classification'] = 'apartment'
            results['confidence'] = min(1.0, final_apartment_score)
        else:
            results['classification'] = 'uncertain_apartment'
            results['confidence'] = final_apartment_score
    else:
        if final_building_score >= confidence_threshold:
            results['classification'] = 'building'
            results['confidence'] = min(1.0, final_building_score)
        else:
            results['classification'] = 'uncertain_building'
            results['confidence'] = final_building_score
    
    # Compile all reasoning
    results['reasoning'] = heating_reasoning + cooling_reasoning + additional_reasoning
    results['scores'] = {
        'apartment': round(final_apartment_score, 3),
        'building': round(final_building_score, 3)
    }
    
    return results


# Example usage and test function
def test_classifier():
    """Test the classifier with example scenarios"""
    
    test_cases = [
        {
            'name': 'Small apartment',
            'heating_kw': 8.5,
            'cooling_kw': 5.2,
            'building_area_m2': 75,
            'building_height_m': 3,
            'estimated_apartments': 1
        },
        {
            'name': 'Medium apartment building',
            'heating_kw': 45,
            'cooling_kw': 28,
            'building_area_m2': 400,
            'building_height_m': 12,
            'estimated_apartments': 6
        },
        {
            'name': 'Large apartment complex',
            'heating_kw': 180,
            'cooling_kw': 120,
            'building_area_m2': 1500,
            'building_height_m': 18,
            'estimated_apartments': 20
        },
        {
            'name': 'Ambiguous case',
            'heating_kw': 15,
            'cooling_kw': 9,
            'building_area_m2': 200,
            'building_height_m': 6,
            'estimated_apartments': 3
        }
    ]
    
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Test Case: {case['name']}")
        print(f"{'='*50}")
        
        result = classify_hvac_capacity(
            case['heating_kw'], 
            case['cooling_kw'],
            case['building_area_m2'],
            case['building_height_m'],
            case['estimated_apartments']
        )
        
        print(f"Input: {case['heating_kw']}kW heating, {case['cooling_kw']}kW cooling")
        print(f"Building: {case['building_area_m2']}m², {case['building_height_m']}m height, ~{case['estimated_apartments']} apartments")
        print(f"\nClassification: {result['classification'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Scores - Apartment: {result['scores']['apartment']:.3f}, Building: {result['scores']['building']:.3f}")
        
        print(f"\nExpected per apartment: {result['metrics']['expected_heating_per_apt']}kW heating, {result['metrics']['expected_cooling_per_apt']}kW cooling")
        print(f"Expected for building: {result['metrics']['expected_building_heating']}kW heating, {result['metrics']['expected_building_cooling']}kW cooling")
        
        print(f"\nReasoning:")
        for reason in result['reasoning']:
            print(f"  • {reason}")

if __name__ == "__main__":
    test_classifier()