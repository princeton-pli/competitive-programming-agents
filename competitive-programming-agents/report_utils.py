def generate_latex_code_for_figures(
    list_of_pictures,
    arrangement,
    caption,
    name,
    star=False,
    position="htb!",
    same_width=False,
    code_between_pics="\\hfill",
    subcaptions=None,
    use_subfigure=True,
    subfigure_pos="t",
    subfigure_width=None,
    use_tabular=False,
):
    """
    arrangement: [row1_pic_count, row2_pic_count, ...]
    subcaptions: list of per-image captions (same length as list_of_pictures) or None
    use_subfigure: wrap each image in a subfigure with its own caption
    subfigure_width: override subfigure width fraction (default: auto-computed from arrangement)
    use_tabular: if True, wrap subfigures in a tabular{c} with \\\\ between rows
    """
    assert len(list_of_pictures) == sum(arrangement), \
        "arrangement does not match number of pictures"
    if subcaptions is not None:
        assert len(subcaptions) == len(list_of_pictures), \
            "subcaptions length must match list_of_pictures"

    env_name = "figure*" if star else "figure"
    figure_src = [f"\\begin{{{env_name}}}[{position}]"]
    figure_src.append("\\centering")

    if use_tabular:
        figure_src.append("\\begin{tabular}{c}")

    # Put images in rows
    idx = 0
    for row_idx, cols in enumerate(arrangement):
        if not use_tabular:
            figure_src.append("\\centering")
        for col in range(cols):
            pic = list_of_pictures[idx]
            subcap = (subcaptions[idx] if subcaptions is not None else None)

            # width fraction for this column
            # keep a little horizontal padding when not using subfigures
            if use_subfigure:
                if subfigure_width is not None:
                    width_frac = subfigure_width
                else:
                    width_frac = (0.96 / (max(arrangement) if same_width else cols))
                figure_src.append(
                    f"\\begin{{subfigure}}[{subfigure_pos}]{{{width_frac:.3f}\\textwidth}}"
                )
                figure_src.append(f"\\includegraphics[width=\\linewidth]{{{pic}}}")
                if subcap is not None:
                    figure_src.append(f"\\caption{{{subcap}}}")
                figure_src.append("\\end{subfigure}")
            else:
                if subfigure_width is not None:
                    width_coef = subfigure_width
                else:
                    width_coef = 0.96 / (max(arrangement) if same_width else cols)
                figure_src.append(
                    f"\\includegraphics[width={width_coef:.3f}\\linewidth]{{{pic}}}"
                )

            if code_between_pics and col != cols - 1:
                figure_src.append(code_between_pics)

            idx += 1

        if row_idx != len(arrangement) - 1:
            if use_tabular:
                figure_src.append("\\\\")
            else:
                figure_src.append("\\bigskip")

    if use_tabular:
        figure_src.append("\\end{tabular}")

    figure_src.append(f"\\caption{{{caption}}}")
    figure_src.append(f"\\label{{fig:{name}}}")
    figure_src.append(f"\\end{{{env_name}}}")
    return "\n".join(figure_src)
