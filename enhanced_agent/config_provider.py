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


if __name__ == "__main__":
    c = EnvConfigProvider()
    connection_str = c.get_db_conn_str()
    print("connection string:")
    print(connection_str)
