import xmltodict

clinvar_sample_path = '/Volumes/xml/sample_xml/RCV000077146.xml'

with open(clinvar_sample_path) as fd:
    doc = xmltodict.parse(fd.read())