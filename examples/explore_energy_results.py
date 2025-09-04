# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

try:
    from ipywidgets import interact, widgets
    from IPython.display import display

    IN_JUPYTER = True
except ImportError:
    IN_JUPYTER = False

# Set paths
mappings_path = r"C:\Users\ufilippi\PycharmProjects\pyBuildingEnergy\examples\archetypes\beat_buildings_batch_2\building_mappings_summary.csv"
results_dir = (
    r"C:\Users\ufilippi\PycharmProjects\pyBuildingEnergy\simulation_results_batch_2"
)

# Load building mappings
try:
    buildings_df = pd.read_csv(mappings_path)
    print(f"Loaded {len(buildings_df)} buildings from mappings file")
except Exception as e:
    print(f"Error loading mappings file: {e}")
    exit(1)


# Function to extract building ID from filename
def get_building_id(filename):
    try:
        return int(Path(filename).stem.split("_")[-1])
    except:
        return None


# Process all result files
energy_data = []
result_files = [
    f
    for f in os.listdir(results_dir)
    if f.startswith("annual_results_") and f.endswith(".csv")
]

for file in result_files:
    building_id = get_building_id(file)
    if building_id is None:
        continue

    file_path = os.path.join(results_dir, file)
    try:
        result = pd.read_csv(file_path)
        if not result.empty:
            # Convert to dict and add building_id
            data = result.iloc[0].to_dict()
            data["building_id"] = building_id
            energy_data.append(data)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Create energy results DataFrame
if energy_data:
    energy_df = pd.DataFrame(energy_data)

    # Merge with building data
    merged_df = pd.merge(buildings_df, energy_df, on="building_id", how="left")

    # Convert to Wh to kWh to avoid confusion with kWh/m²
    merged_df["Q_H_annual"] = merged_df["Q_H_annual"] / 1000
    merged_df["Q_C_annual"] = merged_df["Q_C_annual"] / 1000

    # Convert Wh/m² to kWh/m²
    merged_df["Q_H_annual_per_sqm"] = merged_df["Q_H_annual_per_sqm"] / 1000
    merged_df["Q_C_annual_per_sqm"] = merged_df["Q_C_annual_per_sqm"] / 1000

    # Calculate treated floor area
    merged_df["treated_floor_area"] = merged_df.apply(
        lambda row: (
            row["Q_H_annual"] / row["Q_H_annual_per_sqm"]
            if row["Q_H_annual"] > 0
            else row["Q_C_annual"] / row["Q_C_annual_per_sqm"]
        ),
        axis=1,
    )

    # Save merged data
    output_path = os.path.join(
        os.path.dirname(mappings_path), "building_energy_results.csv"
    )
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

    # Create interactive scatter plot with colored markers
    fig = px.scatter(
        merged_df,
        x="Q_H_annual_per_sqm",
        y="Q_C_annual_per_sqm",
        color="urbem_climate_zone",
        symbol="building_type",
        hover_data=[
            "building_id",
            "urbem_location",
            "building_type",
            "urbem_construction_period",
        ],
        title="Heating vs Cooling Energy Demand by Building Type and Construction Period",
        labels={
            "Q_H_annual_per_sqm": "Heating Demand (kWh/m²)",
            "Q_C_annual_per_sqm": "Cooling Demand (kWh/m²)",
            "urbem_climate_zone": "Climate Zone",
            "building_type": "Building Type",
            "urbem_location": "Location",
            "urbem_construction_period": "Construction Period",
        },
    )

    # Update layout to fix legend overlap and improve appearance
    fig.update_layout(
        title=dict(
            text="<b>Annual Energy Performance</b>",
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
        xaxis_title="Heating Need [kWh/m²]",
        yaxis_title="Cooling Need [kWh/m²]",
        legend=dict(
            y=0.99,
            x=0.99,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=80, r=20, t=80, b=80),
        height=700,
        hovermode="closest",
        hoverlabel=dict(
            font_size=12,
            font_family="Arial",
        ),
        plot_bgcolor="rgba(240,240,240,0.95)",
    )

    # Style markers and hover info
    fig.update_traces(
        marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")),
        hovertemplate=(
            "<b>Building ID</b>: %{customdata[0]}<br>"
            "<b>Location</b>: %{customdata[1]}<br>"
            "<b>Type</b>: %{customdata[2]}<br>"
            "<b>Construction</b>: %{customdata[3]}<br>"
            "<b>Heating</b>: %{x:.1f} kWh/m²<br>"
            "<b>Cooling</b>: %{y:.1f} kWh/m²<br>"
            "<extra></extra>"
        ),
        hoverlabel=dict(
            font_size=12, font_family="Arial", bgcolor=None, bordercolor=None
        ),
    )

    # Add reference lines
    fig.add_hline(
        y=merged_df["Q_C_annual_per_sqm"].median(),
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Median Cooling: {merged_df['Q_C_annual_per_sqm'].median():.1f} kWh/m²",
    )
    fig.add_vline(
        x=merged_df["Q_H_annual_per_sqm"].median(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median Heating: {merged_df['Q_H_annual_per_sqm'].median():.1f} kWh/m²",
    )

    # Show the plot
    if IN_JUPYTER:
        fig.show()
    else:
        # Save the plot to HTML for non-Jupyter environments
        plot_path = os.path.join(
            os.path.dirname(mappings_path), "energy_demand_plot.html"
        )
        fig.write_html(plot_path)
        print(f"\nInteractive plot saved to: {plot_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of buildings: {len(merged_df)}")
    print(
        f"Average Heating Demand: {merged_df['Q_H_annual_per_sqm'].mean():.1f} kWh/m²"
    )
    print(
        f"Average Cooling Demand: {merged_df['Q_C_annual_per_sqm'].mean():.1f} kWh/m²"
    )
    print(f"\nBuilding Types:\n{merged_df['building_type'].value_counts()}")
    print(f"\nClimate Zones:\n{merged_df['urbem_climate_zone'].value_counts()}")

    # Create a function to show detailed results for a specific building
    def show_building_details(building_id):
        try:
            building = merged_df[merged_df["building_id"] == building_id].iloc[0]
            print(f"\nBuilding ID: {building_id}")
            print(
                f"Location: {building['urbem_location']} (Climate Zone: {building['urbem_climate_zone']})"
            )
            print(
                f"Type: {building['building_type']}, Construction: {building['urbem_construction_period']}"
            )
            print(f"Heating Demand: {building['Q_H_annual_per_sqm']:.1f} kWh/m²")
            print(f"Cooling Demand: {building['Q_C_annual_per_sqm']:.1f} kWh/m²")

            # Plot hourly data if available
            hourly_file = os.path.join(results_dir, f"hourly_sim_{building_id}.csv")

            if os.path.exists(hourly_file):
                try:
                    # Read and prepare data
                    hourly_data = pd.read_csv(hourly_file)
                    if not hourly_data.empty:
                        # Handle the unnamed timestamp column (first column)
                        if hourly_data.columns[0] == "Unnamed: 0":
                            hourly_data = hourly_data.rename(
                                columns={"Unnamed: 0": "timestamp"}
                            )

                        # Convert timestamp to datetime
                        if "timestamp" in hourly_data.columns:
                            hourly_data["timestamp"] = pd.to_datetime(
                                hourly_data["timestamp"]
                            )

                        # Create a new figure with secondary y-axis
                        fig = go.Figure()

                        # Add primary y-axis traces (Q_H and Q_C)
                        if all(col in hourly_data.columns for col in ["Q_H", "Q_C"]):
                            fig.add_trace(
                                go.Scatter(
                                    x=hourly_data["timestamp"],
                                    y=hourly_data["Q_H"],
                                    name="Heating Load (W)",
                                    line=dict(color="red", width=1.5),
                                    yaxis="y1",
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=hourly_data["timestamp"],
                                    y=hourly_data["Q_C"],
                                    name="Cooling Load (W)",
                                    line=dict(color="blue", width=1.5),
                                    yaxis="y1",
                                )
                            )

                        # Add secondary y-axis traces (T_op and T_ext) if available
                        if all(col in hourly_data.columns for col in ["T_op", "T_ext"]):
                            fig.add_trace(
                                go.Scatter(
                                    x=hourly_data["timestamp"],
                                    y=hourly_data["T_op"],
                                    name="Operative Temp (°C)",
                                    line=dict(color="green", width=1.5, dash="dot"),
                                    yaxis="y2",
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=hourly_data["timestamp"],
                                    y=hourly_data["T_ext"],
                                    name="Outdoor Temp (°C)",
                                    line=dict(color="purple", width=1.5, dash="dot"),
                                    yaxis="y2",
                                )
                            )

                    # Update layout with both y-axes and fix legend overlap
                    fig.update_layout(
                        title=dict(
                            text=f"<b>Hourly Performance - Building {building_id}</b>",
                            y=0.95,
                            x=0.5,
                            xanchor="center",
                            yanchor="top",
                        ),
                        yaxis2=dict(
                            title="<b>Temperature (°C)</b>",
                            title_font=dict(color="green"),
                            tickfont=dict(color="green"),
                            anchor="x",
                            overlaying="y",
                            side="right",
                        ),
                        legend=dict(
                            title=dict(text="<b>Legend</b>", font=dict(size=11)),
                            orientation="v",
                            y=1.1,
                            yanchor="top",
                            x=1.05,
                            xanchor="left",
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="lightgray",
                            borderwidth=1,
                            font=dict(size=10),
                        ),
                        height=600,
                        margin=dict(l=60, r=150, t=100, b=80),
                        plot_bgcolor="rgba(240,240,240,0.95)",
                    )

                    # Display the plot
                    if IN_JUPYTER:
                        from IPython.display import display, clear_output

                        # Display the plot
                        if IN_JUPYTER:
                            from IPython.display import display, clear_output

                            clear_output(wait=True)
                            display(fig)
                            return fig
                        else:
                            plot_path = os.path.join(
                                os.path.dirname(mappings_path),
                                f"building_{building_id}_hourly_plot.html",
                            )
                            fig.write_html(plot_path)
                            print(f"\nInteractive plot saved to: {plot_path}")
                            return None

                except Exception as e:
                    print(f"Error processing hourly data: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    return None
            else:
                print(f"No hourly data file found for building {building_id}")
                return None

        except Exception as e:
            print(f"Error showing building details: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    if IN_JUPYTER:
        from IPython.display import clear_output

        # Create output widget for the plot
        plot_output = widgets.Output()

        # Create building selector with construction period
        building_selector = widgets.Dropdown(
            options=[
                (
                    f"ID: {bid} - {row['urbem_location']} ({row['urbem_construction_period']})",
                    bid,
                )
                for bid, row in merged_df.set_index("building_id").iterrows()
            ],
            description="Select Building:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="600px"),
        )

        # Define the update function
        def on_building_change(change):
            with plot_output:
                clear_output(wait=True)
                show_building_details(change["new"])

        # Set up the observer
        building_selector.observe(on_building_change, names="value")

        # Display the widgets
        display(building_selector)
        display(plot_output)

        # Show the first building by default
        if len(merged_df) > 0:
            on_building_change({"new": merged_df["building_id"].iloc[0]})
    else:
        # Simple command-line interface for non-Jupyter environments
        print("\nAvailable buildings:")
        for bid in merged_df["building_id"]:
            loc = merged_df[merged_df["building_id"] == bid]["urbem_location"].iloc[0]
            print(f"{bid}: {loc}")

        while True:
            try:
                bid = input("\nEnter building ID to view details (or 'q' to quit): ")
                if bid.lower() == "q":
                    break
                bid = int(bid)
                if bid in merged_df["building_id"].values:
                    show_building_details(bid)
                else:
                    print("Invalid building ID. Please try again.")
            except ValueError:
                print("Please enter a valid building ID or 'q' to quit.")

else:
    print("No energy data found to process")

# %%
