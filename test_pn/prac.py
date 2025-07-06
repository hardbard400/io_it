from grobid_client_python.grobid_client import GrobidClient

client = GrobidClient()
client.process("processFulltextDocument", "Grade-12-Mathematics-Textbook.pdf ", output="tei.xml")
