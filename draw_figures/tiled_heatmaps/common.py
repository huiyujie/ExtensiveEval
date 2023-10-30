import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, patches
from functools import reduce
import numpy as np
from scipy.stats import spearmanr
from itertools import product, combinations, permutations
from datetime import date
from random import sample
import pickle, os, copy


def matrix_spearman(matrix, sample_step=2):
    y_len = len(matrix)
    x_len = len(matrix[0])

    dist_vec = []
    tput_diff_vec = []
    cords = []

    for x in range(0, x_len, sample_step):
        for y in range(0, y_len, sample_step):
            if matrix.mask[y][x]:
                continue
            cords.append((x, y))

    for cord1, cord2 in combinations(cords, 2):
        dist_vec.append(abs(cord1[0] - cord2[0]) + abs(cord1[1] - cord2[1]))
        tputs_diff = abs(
            matrix.data[cord1[1]][cord1[0]] - matrix.data[cord2[1]][cord2[0]]
        )
        tput_diff_vec.append(tputs_diff)

    return abs(spearmanr(dist_vec, tput_diff_vec).correlation)


def gen_tput_matrix(
    df, order, column_names_exclude, tput_col_name="tput", invalid_value=None
):
    column_names = df.columns.to_list()
    for col in column_names_exclude[::]:
        column_names.remove(col)
    col_val = []
    for col in column_names[::]:
        val_one_col = df.loc[:, col].unique().tolist()
        if len(val_one_col) == 1:
            column_names.remove(col)
            continue
        col_val.append(sorted(val_one_col))

    columns = {k: v for (k, v) in zip(column_names, col_val)}
    df_columns_tup = tuple(columns.items())

    tputs_matrix, col_names, col_values, row_names, row_values = ([] for _ in range(5))
    irow, icol, invalid_count = 0, 0, 0
    for i in order:
        col = df_columns_tup[i]
        if i % 2 == 0:
            col_names.append(col[0])
            col_values.append(col[1])
        else:
            row_names.append(col[0])
            row_values.append(col[1])
    for col_info in product(*col_values):
        row = []
        icol = 0
        cond1 = df[col_names[0]] == col_info[0]
        for k, v in zip(col_names, col_info):
            # print((k,v))
            cond1 &= df[k] == v
        for row_info in product(*row_values):
            cond2 = copy.deepcopy(cond1)
            for k, v in zip(row_names, row_info):
                # print((k,v))
                cond2 &= df[k] == v
            result = df.loc[cond2, tput_col_name]
            assert result.size <= 1
            if result.size < 1:
                if not invalid_value:
                    result = np.nan
                else:
                    result = invalid_value
                invalid_count += 1
            else:
                result = result.iloc[0]
            row.append(result)
            icol += 1
        tputs_matrix.append(row)
        irow += 1

    if not invalid_value:
        tputs_masked = np.ma.masked_invalid(tputs_matrix)
    else:
        tputs_masked = np.ma.masked_values(tputs_matrix, invalid_value)

    return tputs_masked, col_names, col_values, row_names, row_values


def scan_order(df, column_names_exclude, tput_col_name="tput", invalid_value=None):
    column_names = df.columns.to_list()
    for col in column_names_exclude[::]:
        column_names.remove(col)

    col_val = []
    for col in column_names[::]:
        val_one_col = df.loc[:, col].unique().tolist()
        if len(val_one_col) == 1:
            column_names.remove(col)
            continue
        col_val.append(sorted(val_one_col))

    # print(col_val)
    total_dim = len(col_val)
    perm_total_dim = list(permutations(range(total_dim)))
    # print(perm_total_dim)
    if total_dim == 6:
        order_to_scan_all = perm_total_dim[: len(perm_total_dim) // 2]
        order_to_scan = sample(order_to_scan_all, len(perm_total_dim) // 10)
    elif total_dim == 5:
        order_to_scan_all = perm_total_dim[: len(perm_total_dim) // 2]
        order_to_scan = sample(order_to_scan_all, len(perm_total_dim) // 5)
    else:
        order_to_scan = perm_total_dim
    # ------------------------------------------------------------------------------------
    spear_results = []
    for order in order_to_scan:
        tputs_masked, _, _, _, _ = gen_tput_matrix(
            df, order, column_names_exclude, tput_col_name, invalid_value
        )
        spear_coef = matrix_spearman(tputs_masked, sample_step=1)
        spear_results.append(spear_coef)

    # print(spear_results)
    # print(order_to_scan)
    return spear_results, order_to_scan


def gen_cmap(
    df,
    fig_name,
    column_names_exclude,
    rep_dict,
    y_subax_loc,
    x_subax_loc,
    PAPER,
    tput_col_name="tput",
    invalid_value=None,
    crange=None,
    cmap_centerd_at=None,
    fontsize=18,
    cbar_label="TP per server thread(OP/sec)",
):
    spear_results, order_to_scan = scan_order(
        df, column_names_exclude, tput_col_name, invalid_value
    )
    order_max = order_to_scan[spear_results.index(max(spear_results))]
    # print(order_max)
    # order_max = (0,3,2,1)
    tputs_masked, col_names, col_values, row_names, row_values = gen_tput_matrix(
        df, order_max, column_names_exclude, tput_col_name, invalid_value
    )
    x_values = row_values[::-1]
    y_values = col_values[::-1]
    y_names = [rep_dict.get(name) for name in col_names[::-1]]
    x_names = [rep_dict.get(name) for name in row_names[::-1]]

    pickle.dump(
        (tputs_masked, x_values, y_values, col_values, row_values, x_names, y_names),
        open("./cache/{}/{}.p".format(PAPER, fig_name), "wb"),
        protocol=-1,
    )
    plot_from_cached(
        fig_name,
        y_subax_loc,
        x_subax_loc,
        PAPER,
        crange,
        cmap_centerd_at,
        fontsize,
        cbar_label,
    )


def format_list_values(vlist):
    fmt_vals = []
    for vals in vlist:
        may_short = True
        if isinstance(vals[0], str):
            may_short = False
            fmt_vals.append(",".join(vals))
            continue

        valsf = []
        if any([v == 0 for v in vals]) or len(vals) <= 5:
            may_short = False

        if vals == [0, 20, 40, 60, 80, 100]:
            fmt_vals.append("0,20...100")
            continue
        elif vals == list(range(0, 110, 10)):
            fmt_vals.append("0,10...100")
            continue
        elif vals[0] == 100 and vals[-1] == 4000:
            fmt_vals.append("100,200,500,1k,2k,4k")
            continue

        if may_short:
            init_ratio = vals[1] / vals[0]

        for idx in range(len(vals)):
            if may_short and idx > 1:
                may_short &= vals[idx] / vals[idx - 1] == init_ratio
            if vals[idx] < 1:
                valsf.append(str(vals[idx]))
            elif vals[idx] >= 1e6:
                valsf.append(str(int(vals[idx] / 1e6)) + "M")
            elif vals[idx] >= 1e4:
                valsf.append(str(int(vals[idx] / 1e3)) + "k")
            else:
                valsf.append(str(int(vals[idx])))

        if may_short:
            short_valsf = valsf[0:3] + ["..."] + [valsf[-1]]
            fmt_vals.append(",".join(short_valsf))
        else:
            fmt_vals.append(",".join(valsf))

    return fmt_vals


def fmt_ant(name, val):
    # return ""
    return "{}:({})...".format(name, val)


def fmt_cbar(vals):
    # max_tick_places = int(np.log10(cbar_ticks[-1]))
    res = []
    for val in vals:
        if val >= 1e6:
            res.append(np.format_float_positional(val / 1e6, 1) + "M")
        elif val >= 1e3:
            res.append(np.format_float_positional(val / 1e3, 1) + "k")
        else:
            res.append(np.format_float_positional(val, 1))
    return res


def plot_from_cached(
    fig_name,
    y_subax_loc,
    x_subax_loc,
    PAPER,
    crange=None,
    cmap_centerd_at=None,
    fontsize=18,
    cbar_label="TP (in ops/sec, white means missing)",
):
    tputs_masked, _, _, col_values, row_values, x_names, y_names = pickle.load(
        open("./cache/{}/{}.p".format(PAPER, fig_name), "rb")
    )

    y_levels_count = len(col_values)
    x_levels_count = len(row_values)

    Y = np.arange(reduce(lambda x, y: x * len(y), col_values, 1))
    X = np.arange(reduce(lambda x, y: x * len(y), row_values, 1))

    row_values_txt = format_list_values(row_values)
    col_values_txt = format_list_values(col_values)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    cmap = plt.get_cmap("gist_yarg").copy()
    cmap.set_bad("white", 1)
    tput_min = tputs_masked.compressed().min()
    tput_max = tputs_masked.compressed().max()

    if tput_min < 0:
        tputs_orig = tputs_masked.data
        tput_min = tputs_orig[tputs_orig > 0].min()

    if crange:
        cmap_centerd_at = sum(crange) / 2
    else:
        cmap_centerd_at = (tput_min + tput_max) / 2

    if not crange:
        cmap_min = tput_min - 0.5 * (cmap_centerd_at - tput_min)
        cmap_max = tput_max + 0.1 * (tput_max - cmap_centerd_at)
    else:
        cmap_min = crange[0]
        cmap_max = crange[1]

    print([cmap_min, cmap_centerd_at, cmap_max])

    cmap_norm = colors.TwoSlopeNorm(
        vmin=cmap_min, vcenter=float(cmap_centerd_at), vmax=cmap_max
    )
    im = ax.pcolormesh(X, Y, tputs_masked, shading="nearest", cmap=cmap, norm=cmap_norm)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not crange:
        cbar_bound = np.linspace(tput_min, tput_max)
    else:
        cbar_bound = np.linspace(cmap_min, cmap_max)

    cbar = fig.colorbar(im, ax=ax, location="right", pad=0.05, boundaries=cbar_bound)
    # cbar.set_label(label=cbar_label,size=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
    cbar_ticks = [cbar_bound[0]] + cbar.get_ticks()
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(fmt_cbar(cbar_ticks))

    x_names.reverse()
    y_names.reverse()
    row_values_txt.reverse()
    col_values_txt.reverse()

    arrow_props = {"arrowstyle": "-", "color": "black", "relpos": (0.5, 0.5)}

    ax_frac = [0.05, 0.1, 0.15]
    x_subax_loc = [-x * len(Y) - 0.5 for x in ax_frac]
    y_subax_loc = [-x * len(X) - 0.5 for x in ax_frac]

    annt_txt = fmt_ant(x_names[-1], row_values_txt[0])
    # annt_txt = "{} : {}...".format(x_names[-1],", ".join(row_values[0]))
    ax.annotate(
        "",
        xy=(-0.5, x_subax_loc[0]),
        xytext=(0.5, x_subax_loc[0]),
        xycoords="data",
        textcoords="data",
        arrowprops=arrow_props,
        annotation_clip=False,
    )
    ax.text(
        x=0.5 + 0.02 * len(X),
        y=x_subax_loc[0],
        s=annt_txt,
        ha="left",
        va="center",
        fontsize=fontsize,
    )

    annt_txt = fmt_ant(y_names[-1], col_values_txt[0])
    # annt_txt = "{} : {}...".format(y_names[-1],", ".join(col_values[0]))
    ax.annotate(
        "",
        xy=(y_subax_loc[0], -0.5),
        xytext=(y_subax_loc[0], +0.5),
        xycoords="data",
        textcoords="data",
        arrowprops=arrow_props,
        annotation_clip=False,
    )
    ax.text(
        x=y_subax_loc[0],
        y=0.5 + 0.02 * len(Y),
        s=annt_txt,
        ha="center",
        va="baseline",
        fontsize=fontsize,
        rotation="vertical",
    )

    for xidx, xx in enumerate(X):
        for yidx, yy in enumerate(Y):
            if tputs_masked.mask[yidx][xidx]:
                ax.fill_between(
                    [xx - 0.5, xx + 0.5],
                    yy - 0.5,
                    yy + 0.5,
                    hatch="x",
                    fc="none",
                    linewidth=0,
                )

    if y_levels_count > 1:
        y_level1 = list(range(0, len(Y), len(col_values[-1])))
        lstys = [(0, (5, 1))]
        if y_levels_count == 3:
            lstys.insert(0, (0, (1, 1)))
        for yy in y_level1[1:]:
            ax.axhline(yy - 0.5, lw=2, c="black", ls=lstys[0])

        annt_txt = fmt_ant(y_names[-2], col_values_txt[1])
        # annt_txt = "{} : {}...".format(y_names[-2],", ".join(col_values[1]))
        ax.annotate(
            "",
            xy=(y_subax_loc[1], -0.5),
            xytext=(y_subax_loc[1], y_level1[1] - 0.5),
            xycoords="data",
            textcoords="data",
            arrowprops=arrow_props,
            annotation_clip=False,
        )
        ax.text(
            x=y_subax_loc[1],
            y=y_level1[1] - 0.5 + 0.02 * len(Y),
            s=annt_txt,
            ha="center",
            va="baseline",
            fontsize=fontsize,
            rotation="vertical",
        )

        if y_levels_count == 3:
            y_level2 = list(range(0, len(Y), len(col_values[-1]) * len(col_values[-2])))
            for yy in y_level2[1:]:
                ax.axhline(yy - 0.5, lw=2, c="black", ls=lstys[1])

            annt_txt = fmt_ant(y_names[-3], col_values_txt[2])
            # annt_txt = "{} : {}...".format(y_names[-3],", ".join(col_values[2]))
            ax.annotate(
                "",
                xy=(y_subax_loc[2], -0.5),
                xytext=(y_subax_loc[2], y_level2[1] - 0.5),
                xycoords="data",
                textcoords="data",
                arrowprops=arrow_props,
                annotation_clip=False,
            )
            ax.text(
                x=y_subax_loc[2],
                y=y_level2[1] - 0.5 + 0.02 * len(Y),
                s=annt_txt,
                ha="center",
                va="baseline",
                fontsize=fontsize,
                rotation="vertical",
            )
    else:
        ax.annotate(
            " ",
            xy=(y_subax_loc[1], -0.5),
            xycoords="data",
            textcoords="data",
            arrowprops=arrow_props,
            annotation_clip=False,
        )

    if x_levels_count > 1:
        x_level1 = list(range(0, len(X), len(row_values[-1])))
        lstys = [(0, (5, 1))]
        if x_levels_count == 3:
            lstys.insert(0, (0, (1, 1)))
        for xx in x_level1[1:]:
            ax.axvline(xx - 0.5, lw=2, c="black", ls=lstys[0])

        annt_txt = fmt_ant(x_names[-2], row_values_txt[1])
        # annt_txt = "{} : {}...".format(x_names[-2],", ".join(row_values[1]))
        ax.annotate(
            "",
            xy=(-0.5, x_subax_loc[1]),
            xytext=(x_level1[1] - 0.5, x_subax_loc[1]),
            xycoords="data",
            textcoords="data",
            arrowprops=arrow_props,
            annotation_clip=False,
        )
        ax.text(
            x=x_level1[1] - 0.5 + 0.02 * len(X),
            y=x_subax_loc[1],
            s=annt_txt,
            ha="left",
            va="center",
            fontsize=fontsize,
        )

        if x_levels_count == 3:
            x_level2 = list(range(0, len(X), len(row_values[-1]) * len(row_values[-2])))
            for xx in x_level2[1:]:
                ax.axvline(xx - 0.5, lw=2, c="black", ls=lstys[1])

            annt_txt = fmt_ant(x_names[-3], row_values_txt[2])
            # annt_txt = "{} : {}...".format(x_names[-3],", ".join(row_values[2]))
            ax.annotate(
                "",
                xy=(-0.5, x_subax_loc[2]),
                xytext=(x_level2[1] - 0.5, x_subax_loc[2]),
                xycoords="data",
                textcoords="data",
                arrowprops=arrow_props,
                annotation_clip=False,
            )
            ax.text(
                x=x_level2[1] - 0.5 + 0.02 * len(X),
                y=x_subax_loc[2],
                s=annt_txt,
                ha="left",
                va="center",
                fontsize=fontsize,
            )
    else:
        ax.annotate(
            " ",
            xy=(-0.5, x_subax_loc[1]),
            xycoords="data",
            textcoords="data",
            arrowprops=arrow_props,
            annotation_clip=False,
        )

    # print(cbar_ticks)
    fig.savefig(
        "./out/{}/{}.pdf".format(PAPER, fig_name),
        bbox_inches="tight",
        dpi=100,
        format="pdf",
    )

    return fig
