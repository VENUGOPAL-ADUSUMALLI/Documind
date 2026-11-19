import os

# Settings requires DATABASE_URL. Tests never use it (they pass their own SQLite
# engine), so give them a harmless value before the app modules are imported.
os.environ.setdefault("DATABASE_URL", "sqlite://")
