"""Small interactive-selection helpers shared across readers.

Kept generic (no HDF5/LAPD knowledge): pick one item from a list of choices at
a terminal / notebook prompt. Readers that hit an ambiguous choice (e.g. several
motion groups in one file) delegate here instead of raising or re-implementing
the ``print list -> read index -> validate`` loop.
"""


def choose_from_list(items, label=lambda x: x, prompt="Index", header=None):
    """Prompt the user to pick one item from ``items`` by index.

    Prints a numbered ``[i] <label(item)>`` list and reads the chosen index from
    stdin, re-prompting on non-numeric or out-of-range input. Returns the chosen
    item (not its index).

    Parameters
    ----------
    items : sequence
        The choices; must be non-empty.
    label : callable, optional
        Maps an item to its display string (default: the item itself).
    prompt : str, optional
        Text before the ``[0-N]`` index request.
    header : str, optional
        A line printed above the list (e.g. what is being chosen).

    Notes
    -----
    Uses :func:`input`, so it blocks in non-interactive/batch contexts. Callers
    that may run headless should offer an explicit override to skip the prompt.
    """
    items = list(items)
    if not items:
        raise ValueError("choose_from_list: no items to choose from.")
    if header:
        print(header)
    for i, item in enumerate(items):
        print(f'  [{i}] {label(item)}')
    while True:
        reply = input(f"{prompt} [0-{len(items) - 1}]: ").strip()
        try:
            idx = int(reply)
        except ValueError:
            print(f"  Not a number: {reply!r}")
            continue
        if 0 <= idx < len(items):
            return items[idx]
        print(f"  Out of range: {idx}")
