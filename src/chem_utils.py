import re 
from IPython.display import SVG
def image_bg_change(svg_output):
    if hasattr(svg_output, 'data'):
        svg_data = svg_output.data
    else:
        svg_data = svg_output
    if isinstance(svg_data, str):
        # Replace existing white fills
        svg_data = re.sub(r'fill:\s*#ffffff', 'fill:#E5ECF6', svg_data, flags=re.IGNORECASE)
        svg_data = re.sub(r'fill:\s*white', 'fill:#E5ECF6', svg_data, flags=re.IGNORECASE)
        svg_data = re.sub(r'rect style="opacity:1.0;fill:\s*#ffffff;stroke:none"', 'rect style="opacity:1.0;fill:#E5ECF6;stroke:none"', svg_data, flags=re.IGNORECASE)
        
        # Inject global background rect
        bg_rect = '<rect width="100%" height="100%" fill="#E5ECF6"/>'
        if "<!-- END OF HEADER -->" in svg_data:
            svg_data = svg_data.replace("<!-- END OF HEADER -->", f"<!-- END OF HEADER -->\n{bg_rect}")
        else:
            svg_data = re.sub(r'(<svg[^>]*>)', lambda m: m.group(1) + f'\n{bg_rect}', svg_data, count=1)
    return SVG(svg_data)