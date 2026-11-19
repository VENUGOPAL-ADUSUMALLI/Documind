from app.config import Settings


def test_railway_style_url_gets_the_psycopg_driver():
    settings = Settings(database_url="postgresql://user:pass@host:5432/db")
    assert settings.database_url == "postgresql+psycopg://user:pass@host:5432/db"


def test_old_postgres_scheme_is_also_converted():
    settings = Settings(database_url="postgres://user:pass@host:5432/db")
    assert settings.database_url == "postgresql+psycopg://user:pass@host:5432/db"


def test_urls_that_are_already_correct_are_left_alone():
    psycopg_url = "postgresql+psycopg://user:pass@host:5432/db"
    assert Settings(database_url=psycopg_url).database_url == psycopg_url
    assert Settings(database_url="sqlite:///x.db").database_url == "sqlite:///x.db"
