# collection/

Вихідні книги WoS + Scopus (у пакет не входять). Покладіть сюди файли з іменами

    Дедублікація_WoS_Scopus_<YYYY>.xlsx     (по одному на рік, напр. 2015–2025)

і побудуйте набір:

    pip install -e ".[ingest]"
    horizon collection-build --path collection --out datasets/wos-scopus
