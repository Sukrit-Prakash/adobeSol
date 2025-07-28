import sys
from pathlib import Path
import typer
from concurrent.futures import ThreadPoolExecutor
from extractor import OutlineExtractor

app = typer.Typer(help="Extract PDF outlines to JSON")

@ app.command()
def extract(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Input directory of PDFs"),
    output_dir: Path = typer.Argument(..., file_okay=False, help="Output directory for JSONs"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = list(input_dir.glob("*.pdf"))
    def _process(pdf: Path):
        json_data = OutlineExtractor(str(pdf)).extract()
        out_path = output_dir / (pdf.stem + ".json")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(typer.style(typer.json(json_data), fg=typer.colors.GREEN))
        typer.echo(f"Processed: {pdf.name} -> {out_path.name}")

    with ThreadPoolExecutor(max_workers=workers) as exe:
        list(exe.map(_process, pdfs))

if __name__ == "__main__":
    app()