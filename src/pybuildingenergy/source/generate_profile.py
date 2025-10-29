import numpy as np
import pandas as pd
from datetime import datetime
import holidays
import requests
import plotly.express as px
import plotly.graph_objects as go

def get_country_code_from_latlon(lat, lon, default='IT'):
    """
    Perform a reverse geocode (using e.g. OpenCage or other API) to fetch
    the ISO 3166-1 alpha-2 country code for the given lat,lon.
    If it fails, returns default.
    """
    # Example using OpenCage Data API (you’ll need an API key)
    api_key = "YOUR_OPENCAGE_API_KEY"
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        'q': f"{lat},{lon}",
        'key': api_key,
        'no_annotations': 1,
        'language': 'en'
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return default
    data = resp.json()
    if not data['results']:
        return default
    components = data['results'][0].get('components', {})
    country_code = components.get('ISO_3166-1_alpha2')
    if country_code:
        return country_code.upper()
    return default

class HourlyProfileGenerator:
    """
    Generate an hourly DataFrame (13 months by default) with working day and holiday flags
    and profile columns for each category: ventilation, heating, cooling,
    occupancy, lighting, appliances.

    Parameters
    ----------
    country : str
        Country code for national holidays (e.g., 'IT', 'US', 'GB', 'FR', 'DE').
    num_months : int
        Number of months to generate (default: 13).
    start_year : int | None
        Starting year; if None, starts from December of the previous year to the current year.
    working_day_profile : np.array(24) | None
        (Retro-compatibilità) profilo 24h usato se non si forniscono profili per categoria.
    holiday_profile : np.array(24) | None
        (Retro-compatibilità) profilo 24h usato se non si forniscono profili per categoria.
    category_profiles : dict | None
        Dictionary with keys in categories:
        {'ventilation','heating','cooling','occupancy','lighting','appliances'}.
        Each value can be:
          - np.array/list of 24 values → used for weekdays and holidays
          - tuple/list of two arrays of 24 → (weekday, holiday)
          - dict with keys 'weekday' and 'holiday' (each array of 24)

        Example:
        category_profiles = {
            'ventilation': {'weekday': v_wd, 'holiday': v_hd},
            'heating': (h_wd, h_hd),
            'lighting': L_24h,               # same for weekdays/holidays
        }
    """

    CATEGORIES = ("ventilation", "heating", "cooling", "occupancy", "lighting", "appliances")

    def __init__(self,
                 country='IT',
                 num_months=13,
                 start_year=None,
                 working_day_profile=None,
                 holiday_profile=None,
                 category_profiles=None):
        self.country = country
        self.num_months = num_months

        # Start year
        current_date = datetime.now()
        self.current_year = current_date.year
        self.start_year = start_year if start_year else self.current_year - 1
        self.start_date = f"{self.start_year}-12-01"

        # Default "storici" (for retro-compatibility)
        default_working = np.array(
            [1,1,1,1,1,1,0.5,0.5,0.5,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8,0.8,1,1],
            dtype=float
        )
        default_holiday = np.array(
            [1,1,1,1,1,1,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,1,1],
            dtype=float
        )

        # If passed, use them instead of defaults
        if working_day_profile is not None:
            self._validate_24(working_day_profile, "working_day_profile")
            default_working = np.asarray(working_day_profile, dtype=float)
        if holiday_profile is not None:
            self._validate_24(holiday_profile, "holiday_profile")
            default_holiday = np.asarray(holiday_profile, dtype=float)

        # Build profiles for categories
        self.profiles = self._build_category_profiles(category_profiles, default_working, default_holiday)

        self.df = None
        self.country_holidays = None

    # ------------------------- helpers -------------------------
    @staticmethod
    def _validate_24(arr, name):
        arr = np.asarray(arr)
        if arr.shape != (24,):
            raise ValueError(f"{name}: must have exactly 24 values (shape {arr.shape})")

    def _coerce_pair(self, value, cat_name):
        """
        Normalizes 'value' into a pair (weekday, holiday), each np.array(24).
        Accepts:
          - array/list 24 → (arr, arr)
          - (arr_wd, arr_hd) → two arrays of 24
          - {'weekday': arr_wd, 'holiday': arr_hd}
        """
        if isinstance(value, dict):
            if "weekday" not in value or "holiday" not in value:
                raise ValueError(f"{cat_name}: dict deve contenere 'weekday' e 'holiday'")
            wd, hd = value["weekday"], value["holiday"]
            self._validate_24(wd, f"{cat_name}.weekday")
            self._validate_24(hd, f"{cat_name}.holiday")
            return np.asarray(wd, dtype=float), np.asarray(hd, dtype=float)

        if isinstance(value, (list, tuple)) and len(value) == 2 and (
            np.asarray(value[0]).shape == (24,) and np.asarray(value[1]).shape == (24,)
        ):
            wd, hd = value
            self._validate_24(wd, f"{cat_name}[0]")
            self._validate_24(hd, f"{cat_name}[1]")
            return np.asarray(wd, dtype=float), np.asarray(hd, dtype=float)

        # single array of 24 → copy to both
        self._validate_24(value, f"{cat_name}")
        arr = np.asarray(value, dtype=float)
        return arr, arr

    def _build_category_profiles(self, category_profiles, default_working, default_holiday):
        """
        Restituisce un dict:
          cat -> {'weekday': np.array(24), 'holiday': np.array(24)}
        """
        profiles = {}

        # If no `category_profiles` is provided, use defaults for all categories
        if category_profiles is None:
            for c in self.CATEGORIES:
                profiles[c] = {"weekday": default_working.copy(), "holiday": default_holiday.copy()}
            return profiles

        # Normalize provided profiles + complete missing categories with defaults
        for c in self.CATEGORIES:
            if c in category_profiles:
                wd, hd = self._coerce_pair(category_profiles[c], c)
                profiles[c] = {"weekday": wd, "holiday": hd}
            else:
                profiles[c] = {"weekday": default_working.copy(), "holiday": default_holiday.copy()}

        return profiles

    # ------------------------- public API -------------------------
    def generate(self):
        """Genera il DataFrame orario con le colonne dei profili per categoria."""
        end_date = pd.Timestamp(self.start_date) + pd.DateOffset(months=self.num_months)
        date_range = pd.date_range(start=self.start_date, end=end_date, freq="h", inclusive="left")

        df = pd.DataFrame({"datetime": date_range})
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Mon ... 6=Sun
        df["day_name"] = df["datetime"].dt.day_name()

        # National holidays (including two years of margin)
        years_needed = list(range(self.start_year, self.start_year + 3))
        self.country_holidays = holidays.country_holidays(self.country, years=years_needed)

        df["is_holiday"] = df["date"].apply(lambda x: x in self.country_holidays)
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
        df["is_working_day"] = ~(df["is_holiday"] | df["is_weekend"])
        df["holiday_name"] = df["date"].apply(lambda x: self.country_holidays.get(x, ""))

        # Category profiles
        for cat, pair in self.profiles.items():
            wd = pair["weekday"]
            hd = pair["holiday"]
            # Apply profile based on day type
            df[f"{cat}_profile"] = df.apply(
                lambda row: wd[row["hour"]] if row["is_working_day"] else hd[row["hour"]],
                axis=1
            )

        # Historical 'profile_value' column for retro-compatibility (uses occupancy)
        df["profile_value"] = df["occupancy_profile"]
        df["day_type"] = df["is_working_day"].apply(lambda x: "Working Day" if x else "Holiday/Weekend")

        self.df = df
        return self.df

    def get_summary(self):
        """Print a brief summary."""
        if self.df is None:
            raise ValueError("DataFrame not generated yet. Call generate() first.")
        print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Start year: {self.start_year} (previous year)")
        print(f"Country: {self.country}")
        print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        print(f"\nTotal hours: {len(self.df)}")
        print(f"Working day hours: {int(self.df['is_working_day'].sum())}")
        print(f"Holiday/Weekend hours: {int((~self.df['is_working_day']).sum())}")
        print("\nAvailable profiles per category:")
        for c in self.CATEGORIES:
            print(f"  - {c}: column '{c}_profile'")


    def plot_annual_profiles(self,
                             categories=None,
                             freq="H",                 # "H" per orario, "D" per media giornaliera
                             include_weekend_shading=True,
                             title="Annual Profiles (Hourly/ Daily)"):
        """
        Crea un grafico Plotly interattivo dei profili sull'intero periodo generato.

        Parametri
        ---------
        categories : list[str] | None
            Quali categorie plottare. Se None, usa tutte quelle disponibili tra:
            ventilation, heating, cooling, occupancy, lighting, appliances,
            e 'internal_gains' se presente come colonna.
        freq : {"H","D"}
            "H" = profilo orario; "D" = media giornaliera (resample).
        include_weekend_shading : bool
            Se True, aggiunge bande verticali per weekend/festivi.
        title : str
            Titolo del grafico.

        Ritorna
        -------
        fig : plotly.graph_objects.Figure
        """
        if self.df is None:
            raise ValueError("DataFrame non generato. Chiama generate() prima di plottare.")

        # Categorie disponibili in base alle colonne presenti
        available = []
        for c in ["ventilation", "heating", "cooling", "occupancy", "lighting", "appliances", "internal_gains"]:
            col = f"{c}_profile"
            if col in self.df.columns:
                available.append(c)

        if categories is None:
            categories = available
        else:
            # tieni solo quelle davvero presenti
            categories = [c for c in categories if c in available]
            if not categories:
                raise ValueError("Nessuna delle categorie richieste è presente nel DataFrame.")

        # Costruisci dataframe lungo per plotly
        cols = [f"{c}_profile" for c in categories]
        plot_df = self.df[["datetime", "is_working_day", "is_weekend", "is_holiday"] + cols].copy()

        if freq.upper() == "D":
            # media giornaliera
            plot_df = (
                plot_df
                .set_index("datetime")
                .resample("D")
                .mean(numeric_only=True)
                .reset_index()
            )
            # dopo il resample le colonne boolean possono diventare float: rigeneriamo flag semplici (weekend/holiday se media>0.5)
            for flag in ["is_working_day", "is_weekend", "is_holiday"]:
                if flag in plot_df.columns:
                    plot_df[flag] = (plot_df[flag] > 0.5).astype(bool)

        # melt
        long_df = plot_df.melt(id_vars=["datetime"], value_vars=cols, var_name="category", value_name="value")
        long_df["category"] = long_df["category"].str.replace("_profile", "", regex=False)

        # line plot
        fig = px.line(
            long_df,
            x="datetime", y="value",
            color="category",
            title=title,
            labels={"value": "profile value (0..1)", "datetime": "Date/Time", "category": "Category"}
        )

        # Opzionale: bande weekend/festivi
        if include_weekend_shading:
            # Usa il df (già orario o giornaliero) per recuperare giorni weekend/festivi
            base = plot_df[["datetime"]].copy()
            # ricava giorno data-only
            base["date"] = base["datetime"].dt.floor("D")
            # prendi un solo record per giorno con le etichette
            day_flags = (self.df[["datetime", "is_weekend", "is_holiday"]]
                         .assign(date=lambda d: d["datetime"].dt.floor("D"))
                         .groupby("date")[["is_weekend", "is_holiday"]].max().reset_index())

            # helper per aggiungere bande
            def _add_bands(mask_col, color_rgba):
                m = day_flags[mask_col].values
                dates = day_flags["date"].values
                for i, is_mask in enumerate(m):
                    if bool(is_mask):
                        start = pd.Timestamp(dates[i])
                        end = start + pd.Timedelta(days=1)
                        fig.add_vrect(x0=start, x1=end,
                                      fillcolor=color_rgba, opacity=0.08, line_width=0)

            # weekend (leggermente grigio), festivi (arancione chiaro)
            _add_bands("is_weekend", "rgba(128,128,128,1)")
            _add_bands("is_holiday", "rgba(255,165,0,1)")

        # layout pulito
        fig.update_layout(
            hovermode="x unified",
            legend_title_text="Category",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.update_yaxes(range=[0, 1], title="Profile value")

        return fig
