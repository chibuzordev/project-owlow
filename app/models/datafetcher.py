# -------------------- DataFetcher --------------------
class DataFetcher:
    """
    Fetch listings from Prop-Intel API with fallbacks.
    - Handles wrapped/unwrapped JSON
    - Handles timeouts, connection errors
    - Can return as DataFrame for downstream use
    """

    def __init__(self, api_url: str, timeout: int = 15):
        self.api_url = api_url
        self.timeout = timeout

    def fetch_json_data(self):
        """
        Fetch JSON from the API endpoint.
        Returns parsed data (list of dicts) or [] if failed.
        """
        try:
            response = requests.get(self.api_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Handle wrapped/unwrapped patterns
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                return data["data"]
            elif isinstance(data, list):
                return data
            else:
                print("[WARN] Unexpected data structure from API")
                return []

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Connection Error: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Request Exception: {err}")

        return []

    def fetch_dataframe(self) -> pd.DataFrame:
        """
        Fetch listings and return as a pandas DataFrame.
        """
        data = self.fetch_json_data()
        if not data:
            print("[ERROR] No data returned from API.")
            return pd.DataFrame()
        return pd.DataFrame(data)
