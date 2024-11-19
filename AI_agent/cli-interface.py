# pdf_agent_cli.py
import click
from pdf_agent.agent import PDFAgent
import json

@click.group()
def cli():
    """PDF AI Agent CLI Interface"""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--analyze', is_flag=True, help='Perform document analysis after loading')
def load(file_path, analyze):
    """Load a PDF document"""
    agent = PDFAgent()
    if agent.load_document(file_path):
        click.echo(f"Successfully loaded document: {file_path}")
        if analyze:
            results = agent.analyze_document()
            click.echo("\nDocument Analysis:")
            click.echo(json.dumps(results, indent=2))
    else:
        click.echo("Failed to load document")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.argument('question', type=str)
def ask(file_path, question):
    """Ask a question about the document"""
    agent = PDFAgent()
    if agent.load_document(file_path):
        answer = agent.ask_question(question)
        click.echo("\nAnswer:")
        click.echo(json.dumps(answer, indent=2))

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.argument('query', type=str)
def search(file_path, query):
    """Search for content in the document"""
    agent = PDFAgent()
    if agent.load_document(file_path):
        results = agent.search_content(query)
        click.echo("\nSearch Results:")
        click.echo(json.dumps(results, indent=2))

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output-dir', type=click.Path(), default='extracted_images',
              help='Directory to save extracted images')
def extract_images(file_path, output_dir):
    """Extract images from the document"""
    import os
    
    agent = PDFAgent()
    if agent.load_document(file_path):
        images = agent.extract_images()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for i, img in enumerate(images):
            output_path = os.path.join(output_dir, f"image_{i}.{img['extension']}")
            with open(output_path, 'wb') as f:
                f.write(img['bytes'])
            
            # Perform OCR if image contains text
            text = agent.get_text_from_image(img['bytes'])
            if text.strip():
                click.echo(f"\nText found in image {i}:")
                click.echo(text)
                
        click.echo(f"\nExtracted {len(images)} images to {output_dir}")

if __name__ == '__main__':
    cli()
