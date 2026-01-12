import os
from dotenv import load_dotenv


class EnvConfigProvider:
    def __init__(self):
        load_dotenv()

    def get_openai_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set")
        return api_key

    def get_internal_auth_token(self) -> str:
        internal_token = os.getenv("INTERNAL_AUTH_TOKEN")
        if not internal_token:
            raise EnvironmentError(
                "Internal auth token variable is not set")
        return internal_token

    def get_db_conn_str(self) -> str:
        value = os.getenv("USE_CRM_ADVANCE_DB", "").strip().lower()
        if value == "true":
            return self._get_crm_advance_conn_str()
        return self._get_foundation_conn_str()

    def _get_crm_advance_conn_str(self) -> str:
        """
        CRM_Advance, on-prem SQL Server, Windows authentication.
        Intended for running on a domain-joined Windows machine
        where your current Windows identity has access.
        """
        db_driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
        db_host = os.getenv(
            "DB_HOST")
        db_name = os.getenv("DB_NAME", "CRM_Advance")
        db_encrypt = os.getenv("DB_ENCRYPT", "yes")
        db_trust_cert = os.getenv("DB_TRUST_CERT", "yes")
        db_auth = os.getenv("DB_AUTH", "windows").strip().lower()

        if db_auth != "windows":
            raise RuntimeError(
                "DB_AUTH must be 'windows' for CRM_Advance configuration"
            )

        conn_str = (
            f"DRIVER={{{db_driver}}};"
            f"SERVER={db_host};"
            f"DATABASE={db_name};"
            "Trusted_Connection=yes;"
            f"Encrypt={db_encrypt};"
            f"TrustServerCertificate={db_trust_cert};"
            "Connection Timeout=15;"
        )
        return conn_str

    def _get_foundation_conn_str(self) -> str:
        """
        Foundation website DB (RDS SQL Server), SQL authentication.
        Local dev: usually 127.0.0.1:1433 via SSM tunnel.
        EC2/prod: FOUNDATION_DB_HOST = RDS endpoint.
        """
        db_driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
        db_host = os.getenv("FOUNDATION_DB_HOST", "127.0.0.1")
        db_port = int(os.getenv("FOUNDATION_DB_HOST_PORT", "1433"))
        db_name = os.getenv("FOUNDATION_DB_NAME", "fnd_website")
        db_user = os.getenv("FOUNDATION_DB_USER")
        db_password = os.getenv("FOUNDATION_DB_PASSWORD")

        if not all([db_user, db_password, db_host]):
            raise RuntimeError(
                "Foundation DB environment variables are not fully set "
                "(FOUNDATION_DB_USER, FOUNDATION_DB_PASSWORD, FOUNDATION_DB_HOST)"
            )

        conn_str = (
            f"DRIVER={{{db_driver}}};"
            f"SERVER={db_host},{db_port};"
            f"DATABASE={db_name};"
            f"UID={db_user};"
            f"PWD={db_password};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )
        return conn_str


if __name__ == "__main__":
    c = EnvConfigProvider()
    connection_str = c.get_db_conn_str()
    print("connection string:")
    print(connection_str)
