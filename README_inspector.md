# Parquet Inspector

Lokální read-only prohlížeč velkého souboru `train.parquet`. Data čte stránkovaně přes DuckDB, takže se celý 2,9GB dataset nenačítá do paměti.

```bash
pip install -r requirements.txt
python parquet_inspector.py
```

Pak otevřete <http://127.0.0.1:8765>. Jiný soubor lze načíst pomocí `python parquet_inspector.py --file cesta/data.parquet`.

Kliknutí na řádek otevře celý záznam a vykreslí MS/MS spektrum. Kliknutí na záhlaví sloupce řadí data; hledání lze omezit na konkrétní sloupec.
