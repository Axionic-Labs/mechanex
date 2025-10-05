from typing import List
from .base import _BaseModule

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from itertools import cycle
import numpy as np

def truncate_text(text, max_width, font_properties):
    """Truncates text with '...' if it exceeds the max_width."""
    char_limit = int(max_width * 10)
    if len(text) > char_limit:
        return text[:char_limit-3] + "..."
    return text

def add_glow_effect(patch, ax, glow_color, n_layers=10, max_alpha=0.3, diff_linewidth=1.5):
    """Adds a glow effect to a matplotlib patch."""
    patch_glow = []
    for i in range(n_layers):
        p = type(patch)((patch.get_bbox().x0, patch.get_bbox().y0), patch.get_width(), patch.get_height(),
                        boxstyle=patch.get_boxstyle(),
                        mutation_scale=patch.get_mutation_scale(),
                        mutation_aspect=patch.get_mutation_aspect(),
                        edgecolor=to_rgba(glow_color, alpha=0),
                        facecolor=to_rgba(glow_color, alpha=max_alpha * (1 - i / n_layers)**2),
                        linewidth=(patch.get_linewidth() + diff_linewidth * (i / n_layers)))
        patch_glow.append(p)
    return patch_glow

class RAAGModule(_BaseModule):
    """Module for Retrieval-Augmented Answer Generation APIs."""
    def generate(self, qa_entries: List[dict], docs: List[dict]) -> dict:
        """
        Performs Retrieval-Augmented Answer Generation.
        Corresponds to the /raag/generate endpoint.
        """
        raag = self._post("/raag/generate", {"qa_entries": qa_entries, "docs": docs})
        try:
            data = raag["data_by_question"]["0"]
            edges_dict = data["validated"]["validated_edges"]
            chunks = data["bundle"]["chunk_spans"]
            answer_tokens = raag["answer_tokens"][0]
            prompt_tokens = raag["prompt_tokens"][0]
        except NameError:
            print("Variable 'raag' not found.")
        prompt_len = len(prompt_tokens)

        # --- Style and Theme Configuration (Red Palette) ---
        BG_COLOR = "#1a1a1a"
        TEXT_COLOR = "#EAEAEA"
        ACTIVE_TOKEN_COLOR = "#A00000"  # Darker Red
        INACTIVE_TOKEN_COLOR = "#400000" # Even darker red/maroon
        CHUNK_COLOR = "#CC0000"         # Brighter Red
        #EDGE_COLOR = "#FF6666"         # This is now dynamically assigned
        GLOW_COLOR = "#FFCCCC"          # Pale Red for glow

        # --- MODIFICATION: Create a color map for 'best_layer' ---
        # Find all unique layers to create a color mapping
        all_layers = set()
        for edges in edges_dict.values():
            for e in edges:
                if 'best_layer' in e:
                    all_layers.add(e['best_layer'])
        sorted_layers = sorted(list(all_layers))

        # Create a color palette that cycles if there are more layers than colors
        layer_palette = ['#FFB3BA', '#FF7F7F', '#FF4C4C', '#E60000', '#A00000', '#660000']
        color_cycle = cycle(layer_palette)
        layer_color_map = {layer: next(color_cycle) for layer in sorted_layers}

        answer_token_indices_with_edges = [
            int(k) - prompt_len for k, v in edges_dict.items() if v
        ]
        answer_token_indices_with_edges = [i for i in answer_token_indices_with_edges if 0 <= i < len(answer_tokens)]
        all_answer_indices = list(range(len(answer_tokens)))

        # --- Layout and Spacing ---
        vertical_spacing = 1.2
        left_x = 1.5
        right_x = 10
        node_width_left = 3.0
        node_width_right = 5.0
        node_height = 0.8

        max_items = max(len(all_answer_indices), len(chunks))
        total_height = max_items * vertical_spacing

        left_y = [total_height - i * vertical_spacing for i in range(len(all_answer_indices))]
        right_y = [total_height - i * vertical_spacing for i in range(len(chunks))]

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(16, total_height * 0.35 + 4))
        fig.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Draw left nodes (all answer tokens)
        left_positions = {}
        for i, idx in enumerate(all_answer_indices):
            token = answer_tokens[idx].replace("Ġ", " ").strip()
            y = left_y[i]
            left_positions[idx] = (left_x, y)
            
            facecolor = ACTIVE_TOKEN_COLOR if idx in answer_token_indices_with_edges else INACTIVE_TOKEN_COLOR
            
            bbox = FancyBboxPatch((left_x - node_width_left / 2, y - node_height / 2),
                                width=node_width_left, height=node_height,
                                boxstyle="round,pad=0.1,rounding_size=0.2",
                                fc=facecolor, ec="none", lw=1.5)
            
            if idx in answer_token_indices_with_edges:
                for p_glow in add_glow_effect(bbox, ax, GLOW_COLOR):
                    ax.add_patch(p_glow)

            ax.add_patch(bbox)
            
            display_text = truncate_text(token, node_width_left, {})
            ax.text(left_x, y, display_text, va='center', ha='center', fontsize=11, color=TEXT_COLOR, weight='bold')

        # Draw right nodes (chunks)
        right_positions = {}
        for i, chunk in enumerate(chunks):
            y = right_y[i]
            right_positions[chunk["chunk_global_id"]] = (right_x, y)
            
            chunk_text_preview = "".join(prompt_tokens[chunk["tok_start_in_prompt"]:chunk["tok_end_in_prompt"]])
            chunk_text_preview = chunk_text_preview.replace("Ġ", " ").replace(" ", " ").replace("##", "").strip()
            
            bbox = FancyBboxPatch((right_x - node_width_right / 2, y - node_height / 2),
                                width=node_width_right, height=node_height,
                                boxstyle="round,pad=0.1,rounding_size=0.2",
                                fc=CHUNK_COLOR, ec="none")
            
            for p_glow in add_glow_effect(bbox, ax, GLOW_COLOR, max_alpha=0.2):
                ax.add_patch(p_glow)
                
            ax.add_patch(bbox)
            
            display_text = truncate_text(chunk_text_preview, node_width_right, {})
            ax.text(right_x, y, display_text, va='center', ha='center', fontsize=10, color=TEXT_COLOR)

        for token_idx_str, edges in edges_dict.items():
            token_idx = int(token_idx_str)
            answer_idx = token_idx - prompt_len
            
            if answer_idx in left_positions:
                x0, y0 = left_positions[answer_idx]
                
                for e in edges:
                    chunk_id = e["chunk_global_id"]
                    if chunk_id in right_positions and 'best_layer' in e:
                        x1, y1 = right_positions[chunk_id]
                        
                        # Get the color for the current edge's layer
                        layer = e['best_layer']
                        edge_color = layer_color_map.get(layer, '#FFFFFF') # Default to white if layer not in map
                        
                        # Straight line connection
                        arrow = FancyArrowPatch(
                            (x0 + node_width_left / 2, y0),
                            (x1 - node_width_right / 2, y1),
                            color=to_rgba(edge_color, alpha=0.9), # Use dynamic color
                            lw=max(e['weight'] * 2.5, 0.5),
                            arrowstyle='-', # No arrowhead
                            shrinkA=5, shrinkB=5, # Add padding
                            mutation_scale=1 # No curvature
                        )
                        
                        # Add glow to edges using the dynamic color
                        for i in range(5):
                            glow_arrow = FancyArrowPatch(
                                (x0 + node_width_left / 2, y0),
                                (x1 - node_width_right / 2, y1),
                                color=to_rgba(edge_color, alpha=0.15 * (1 - i / 5)), # Use dynamic color for glow
                                lw=max(e['weight'] * 2.5, 0.5) + (i + 1) * 1.5,
                                arrowstyle='-', shrinkA=5, shrinkB=5,
                                mutation_scale=1
                            )
                            ax.add_patch(glow_arrow)

                        ax.add_patch(arrow)

        # Final adjustments
        ax.set_xlim(left_x - node_width_left, right_x + node_width_right)
        ax.set_ylim(-1.5, total_height + vertical_spacing) # Adjusted ylim for legend
        ax.axis('off')
        plt.title("Answer Token to Document Chunk Attribution", fontsize=18, pad=20, color=TEXT_COLOR, weight='bold')

        legend_elements = [Line2D([0], [0], color=color, lw=4, label=f'Layer {layer}')
                        for layer, color in sorted(layer_color_map.items())]

        ax.legend(handles=legend_elements,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.02),
                ncol=len(legend_elements),
                facecolor=BG_COLOR,
                edgecolor=BG_COLOR,
                labelcolor=TEXT_COLOR,
                fontsize='large',
                frameon=False)

        plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for the legend
        plt.show()

        return raag