import matplotlib.pyplot as plt


def plot_series(ax, series, label):
    ax.plot(series, label=label)
    ax.legend()
    ax.grid(True)


def plot_time_series(
    data,
    title=None,
    xlabel=None,
    ylabel=None,
    figsize=(10, 6),
    legend=None,
    separate_plots=False,
):
    if separate_plots and isinstance(data, list):
        num_series = len(data)
        fig, axes = plt.subplots(num_series, 1, figsize=figsize, sharex=True)

        for i, series in enumerate(data):
            ax = axes if num_series == 1 else axes[i]
            label = (
                f"Series {i + 1}" if legend is None or i >= len(legend) else legend[i]
            )
            plot_series(ax, series, label)
            if i == 0 and title:
                ax.set_title(title)
            if i == num_series - 1:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel[i] if isinstance(ylabel, list) else ylabel)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if isinstance(data, list):
            for i, series in enumerate(data):
                label = (
                    f"Series {i + 1}"
                    if legend is None or i >= len(legend)
                    else legend[i]
                )
                plot_series(ax, series, label)
        else:
            plot_series(ax, data, legend[0] if legend else "Series")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()


def plot_input_output_sequences(
    input_seq,
    output_seq,
    figsize=(10, 8),
    input_labels=None,
    output_labels=None,
    separate_plots=True,
):
    num_input_vars = input_seq.shape[1] if input_seq.ndim > 1 else 1
    num_output_vars = output_seq.shape[1] if output_seq.ndim > 1 else 1

    if separate_plots:
        input_data = [input_seq[:, i] for i in range(num_input_vars)]
        output_data = [output_seq[:, i] for i in range(num_output_vars)]

        plot_time_series(
            input_data,
            title="Input Sequence",
            xlabel="Time Steps",
            ylabel=input_labels,
            figsize=figsize,
            separate_plots=True,
        )
        plot_time_series(
            output_data,
            title="Output Sequence",
            xlabel="Time Steps",
            ylabel=output_labels,
            figsize=figsize,
            separate_plots=True,
        )
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        if num_input_vars == 1:
            ax1.plot(input_seq)
        else:
            for i in range(num_input_vars):
                ax1.plot(
                    input_seq[:, i],
                    label=(
                        f"Input Var {i+1}" if input_labels is None else input_labels[i]
                    ),
                )
            ax1.legend()

        ax1.set_title("Input Sequence")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Value")
        ax1.grid(True)

        if num_output_vars == 1:
            ax2.plot(output_seq)
        else:
            for i in range(num_output_vars):
                ax2.plot(
                    output_seq[:, i],
                    label=(
                        f"Output Var {i+1}"
                        if output_labels is None
                        else output_labels[i]
                    ),
                )
            ax2.legend()

        ax2.set_title("Output Sequence")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Value")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
