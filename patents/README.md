# patents/

Вихідні книги патентного реєстру (у пакет не входять). Файли розпізнаються за шаблоном імені:

    *FinalResult*.xlsx   реєстр (обов'язковий)
    *WIPO*.xlsx          аркуш публікацій WO (необов'язковий)
    *ip5*.xlsx           аркуш заявок IP5 (необов'язковий)

Побудова набору:

    pip install -e ".[ingest]"
    horizon patents-build --path patents --out datasets/patents
