import os
import toml


class Config:
    _instance = None  # Singleton instance

    def __new__(cls, config_file_path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            if config_file_path is None:
                config_file_path = os.path.join('./config.toml')
            cls._instance.config = toml.load(config_file_path)

            # 🔹 FIX: Set cả sections
            for section, values in cls._instance.config.items():
                if isinstance(values, dict):
                    # Set section như một dict attribute
                    setattr(cls._instance, section, values)

                    # CŨNG set từng key từ section (để backward compatible)
                    for key, value in values.items():
                        setattr(cls._instance, key, value)

        return cls._instance

    def __init__(self, config_path="config.toml"):
        print(f"\nDEBUG: Config loaded from {config_path}")
        print(f"DEBUG: config.privacy = {getattr(self, 'privacy', 'ATTRIBUTE NOT FOUND')}")
        print(f"DEBUG: config.enable_privacy_summary = {getattr(self, 'enable_privacy_summary', 'ATTRIBUTE NOT FOUND')}")

class GlobalVar:
    query_number = 0

    @staticmethod
    def set_query_number(num):
        GlobalVar.query_number = num

    @staticmethod
    def get_query_number():
        return GlobalVar.query_number


if __name__ == '__main__':
    cfg = Config()
    print(cfg.test_init_total_number_documents)  # Outputs: 20
    print(cfg.api_key)  # Outputs the API key
    print(f"Privacy section: {cfg.privacy}")  # ← THÊM TEST
