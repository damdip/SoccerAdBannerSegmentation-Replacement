import graphviz
from pathlib import Path

def create_segmentation_architecture_diagram():
    """
    Crea un diagramma dell'architettura SegmentationModel (DINOv2-small + Decoder).
    Il flusso va da sinistra a destra (rankdir=LR).
    """
    dot = graphviz.Digraph(
        name='SegmentationModel',
        comment='SegmentationModel Architecture',
        format='png'
    )
    # Configurazione globale
    dot.attr(
        rankdir='LR',  # Left to Right
        size='16,8',
        dpi='300',
        bgcolor='white',
        fontname='Arial',
        fontsize='14',
        splines='ortho',
        nodesep='0.8',
        ranksep='1.2'
    )

    # Stili
    input_style = {
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': '#E3F2FD',
        'color': '#1976D2',
        'fontcolor': '#1976D2',
        'fontsize': '12',
        'fontname': 'Arial Bold'
    }
    backbone_style = {
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': '#F3E5F5',
        'color': '#7B1FA2',
        'fontcolor': '#7B1FA2',
        'fontsize': '11',
        'fontname': 'Arial'
    }
    component_style = {
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': '#FFF3E0',
        'color': '#F57C00',
        'fontcolor': '#F57C00',
        'fontsize': '10',
        'fontname': 'Arial'
    }
    decoder_style = {
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': '#E8F5E9',
        'color': '#388E3C',
        'fontcolor': '#388E3C',
        'fontsize': '12',
        'fontname': 'Arial Bold'
    }
    output_style = {
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': '#FFEBEE',
        'color': '#D32F2F',
        'fontcolor': '#D32F2F',
        'fontsize': '12',
        'fontname': 'Arial Bold'
    }

    # Output (più a destra)
    dot.node('output',
        'Output Mask\n[1, 2, 224, 224]\n\nBinary Segmentation\n(Background/Banner)',
        **output_style
    )

    # Decoder
    dot.node('upsample',
        'Upsampling\n\nInput: [1, 2, 14, 14]\nOutput: [1, 2, 224, 224]',
        **decoder_style
    )
    dot.node('conv2d_3',
        'Conv2d 128→2 (1x1)\n\nInput: [1, 128, 14, 14]\nOutput: [1, 2, 14, 14]\nParams: 258',
        **component_style
    )
    dot.node('conv2d_2',
        'Conv2d 256→128 (3x3 + ReLU)\n\nInput: [1, 256, 14, 14]\nOutput: [1, 128, 14, 14]\nParams: ~74K',
        **component_style
    )
    dot.node('conv2d_1',
        'Conv2d 384→256 (3x3 + ReLU)\n\nInput: [1, 384, 14, 14]\nOutput: [1, 256, 14, 14]\nParams: ~353K',
        **component_style
    )
    dot.node('reshape',
        'Reshape\n\nInput: [1, 384, 196]\nOutput: [1, 384, 14, 14]',
        **component_style
    )

    # Backbone
    dot.node('dinov2',
        'DINOv2-small\n(Backbone)\n\nInput: [1, 3, 224, 224]\nOutput: [1, 384, 196]\nParams: ~22M',
        **backbone_style
    )

    # Input
    dot.node('input',
        'Input Image\n[1, 3, 224, 224]\n\nRGB Image\n224x224 pixels',
        **input_style
    )

    # Connessioni principali
    edge_style = {
        'color': '#424242',
        'penwidth': '2',
        'arrowsize': '1.2'
    }
    dot.edge('input', 'dinov2', label='  Forward Pass  ', **edge_style)
    dot.edge('dinov2', 'reshape', label='  Feature Embedding  ', **edge_style)
    dot.edge('reshape', 'conv2d_1', label='  Feature Map  ', **edge_style)
    dot.edge('conv2d_1', 'conv2d_2', label='  ', **edge_style)
    dot.edge('conv2d_2', 'conv2d_3', label='  ', **edge_style)
    dot.edge('conv2d_3', 'upsample', label='  ', **edge_style)
    dot.edge('upsample', 'output', label='  ', **edge_style)

    # Raggruppamento Backbone
    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(style='rounded,dashed', color='#7B1FA2', label='Backbone')
        c.node('dinov2')
        c.node('reshape')

    # Raggruppamento Decoder
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(style='rounded,dashed', color='#388E3C', label='Decoder')
        c.node('conv2d_1')
        c.node('conv2d_2')
        c.node('conv2d_3')
        c.node('upsample')

    return dot

def save_architecture_diagram(output_dir='./diagrams', filename='segmentation_architecture'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dot = create_segmentation_architecture_diagram()
    formats = ['png', 'pdf', 'svg']
    saved_files = []
    for fmt in formats:
        output_path = Path(output_dir) / f"{filename}.{fmt}"
        dot.format = fmt
        dot.render(str(output_path.with_suffix('')), cleanup=True)
        saved_files.append(output_path)
        print(f"✅ Diagramma salvato: {output_path}")
    return saved_files

if __name__ == "__main__":
    print("🚀 Generazione diagramma architettura SegmentationModel...")
    try:
        saved_files = save_architecture_diagram()
        print(f"\n🎉 Completato! Generati {len(saved_files)} file:")
        for file_path in saved_files:
            print(f"   📁 {file_path}")
        print(f"\n💡 Usa PNG per presentazioni, PDF per documenti, SVG per web/editing vettoriale")
    except Exception as e:
        print(f"❌ Errore durante la generazione: {str(e)}")
        print("💡 Assicurati di avere graphviz installato: pip install graphviz")