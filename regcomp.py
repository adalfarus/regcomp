import msvcrt
import sys
import os
import re


# Standard typing imports for aps
import collections.abc as _a
import typing as _ty
import types as _ts


def unnest_iterable(iterable: _a.Iterable, max_depth: int = 4) -> list[_ty.Any]:
    """Flatten a nested iterable structure up to a specified depth.

    Args:
        iterable (Any): The nested structure to flatten.
        max_depth (int): Maximum depth to flatten.

    Returns:
        list: A flattened list of elements up to `max_depth`.
    """
    def _lod_helper(curr_lod: list[_ty.Any | list], big_lster: list[_ty.Any], depth: int) -> list[_ty.Any]:
        for x in curr_lod:
            if isinstance(x, list) and depth > 0:
                _lod_helper(x, big_lster, depth - 1)
            else:
                big_lster.append(x)
        return big_lster
    return _lod_helper(list(iterable), [], max_depth)


class c(str):
    ...


def get_gap_of_size(t: int, in_this: list[tuple[int, list[str]]], until_address: int) -> int:
    # Step 1: Prepare the list of used address ranges
    used_ranges = []

    # Gather the used ranges from the input
    for start, labels in in_this:
        size = len(labels)  # The size is the length of the label list
        end = start + size - 1  # Calculate the end address for this range

        # Only consider ranges that are within the address space
        if start <= until_address:
            # Adjust the end if it exceeds the maximum allowed address
            if end > until_address:
                end = until_address
            used_ranges.append((start, end))

    # Step 2: Sort the ranges by their start address
    used_ranges.sort()

    # Step 3: Merge overlapping or adjacent ranges
    merged_ranges = []
    for start, end in used_ranges:
        if merged_ranges and merged_ranges[-1][1] >= start - 1:
            # Merge if there's overlap or adjacency
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    # Step 4: Calculate gaps between consecutive used ranges
    available_gaps = []

    # Check the gap before the first range
    if merged_ranges:
        first_range_start = merged_ranges[0][0]
        if first_range_start > 0 and first_range_start > t:
            available_gaps.append((0, first_range_start))  # The gap before the first used range

    # Check the gaps between consecutive merged ranges
    for i in range(1, len(merged_ranges)):
        prev_end = merged_ranges[i - 1][1]
        curr_start = merged_ranges[i][0]

        # Calculate the gap between the previous end and the current start
        gap = curr_start - prev_end - 1  # Gap is the space between the end of one range and the start of the next
        if gap >= t:
            available_gaps.append((prev_end, curr_start))

    # Step 5: Consider the gap after the last used range (if there's any address space after until_address)
    if merged_ranges:
        last_end = merged_ranges[-1][1]
        remaining_gap = until_address - last_end
        if remaining_gap >= t:
            available_gaps.append((last_end, until_address))

    # Step 6: Find the smallest gap that can accommodate the requested size t
    if available_gaps:
        # Return the smallest gap that can fit the requested size
        return min(available_gaps, key=lambda x: x[1] - x[0])[0] + 1  # Return start address
    elif not in_this:
        return 0
    return -1  # No gap large enough for t found, so we return an error value


def map_to_indexes(max_address, data) -> list:
    # Initialize the array with max_address+1 elements set to None
    array = [None] * (max_address + 1)

    # Iterate over each (start, strs) tuple in data
    for start, strs in data:
        # Assign the strings to the corresponding positions in the array
        for idx, string in enumerate(strs):
            # Start at the given index and map each string to the following indices
            if start + idx <= max_address:  # Ensure we don't exceed the max address
                array[start + idx] = string

    return array


def replace_placeholders_with_whitespace_condition(dat, all_placeholders, lookup_table):
    for placeholder in all_placeholders:
        # Create a pattern that matches each placeholder preceded by a whitespace or start of string,
        # and followed by a whitespace or end of string
        pattern = r'(^|\s)' + re.escape(placeholder) + r'(\s|$)'

        # Replace the matched placeholder with the corresponding value from the lookup_table
        dat = re.sub(pattern,
                     lambda m: m.group(1) + str(lookup_table.get(m.group(0).strip(), m.group(0).strip())) + m.group(2),
                     dat)

    return dat


def main(input: str, output: str) -> None:
    # Works like this:
    # aarg means another arg and is for extra args like
    # sta 10, #2
    # # means literal, it can accept minus and ASCII
    # Using marg instead of arg as the placeholder means it can also take minus.
    # This is reserved for the jumps
    # c() is defined to let the compiler know "I need to change this"
    valid_commands = {
        "lda": ("#{arg}", "{arg}", "({arg})"),
        "sta": ("{arg}", "({arg})", c("{arg} #{aarg}"), c("({arg}) #{aarg}")),
        "add": (c("#{arg} {aarg}"), "{arg}", c("{arg} #{aarg}"), c("#{arg} #{aarg}")),
        "sub": (c("#{arg} {aarg}"), "{arg}", c("{arg} #{aarg}"), c("#{arg} #{aarg}")),
        "mul": (c("#{arg} {aarg}"), "{arg}", c("{arg} #{aarg}"), c("#{arg} #{aarg}")),
        "div": (c("#{arg} {aarg}"), "{arg}", c("{arg} #{aarg}"), c("#{arg} #{aarg}")),  # All these c("#{}") just need an "lda #{}" inserted before them
        "jmp": ("{arg}", "({arg})", c("{marg}"), c("({marg})")),  # All the marg just need to be calculated
        "jnz": ("{arg}", "({arg})", c("{marg}"), c("({marg})")),  # into actual addresses and inserted.
        "jze": ("{arg}", "({arg})", c("{marg}"), c("({marg})")),
        "jle": ("{arg}", "({arg})", c("{marg}"), c("({marg})")),
        "stp": ("",),
        "call": (c("{arg}"),),
        "ret": (c(""),),
    }
    max_address = 199  # So we don't have an invalid address
    fd = fd_length = 0
    try:
        fd = os.open(input, os.O_RDWR | os.O_CREAT)
        fd_length = os.fstat(fd).st_size
        msvcrt.locking(fd, msvcrt.LK_RLCK, fd_length)
        data = os.read(fd, fd_length).decode("utf-8").split("\n")
        os.lseek(fd, 0, 0)  # Go to beginning of file so locking can happen successfully
    except OSError as e:
        print(f"OSError while reading input file: {e}")
        exit(-1)
    finally:
        if fd != 0:
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, fd_length)
            except Exception as e:
                print(f"Error unlocking the file: {e}")
            os.close(fd)

    reserved: list[tuple[int, list[str]]] = []
    labels_in_order: list[tuple[str, list[str]]] = []
    labeled: int = 0  # 0 = false, 1 = reserved, 2 = labels_in_order
    for i, datum in enumerate(data):
        true_datum, comment, *_ = datum.split(";", maxsplit=1) + [""]
        true_datum = true_datum.strip()
        if not datum or not true_datum:
            continue  # Empty line
        elif not true_datum.startswith("."):
            label, *rest = true_datum.split(" ", maxsplit=1)
            if label.isnumeric():
                reserved.append((int(label), rest))
                labeled = 1
            elif label[:-2].isnumeric() and label[-2:] == "ff":
                last_reserved = reserved[-1]
                if last_reserved[0] + len(last_reserved[1]) == int(label[:-2]):
                    last_reserved[1].extend(rest)  # Can actually only ever be one element
                else:
                    raise ValueError(f"FF-Label '{label}' is not ff, it has to be '{last_reserved[0] + len(last_reserved[1])}ff'")
            else:
                if label.endswith(":"):
                    label = label[:-1]  # This adds one empty rest into the rels
                elif not rest:  # rest == []
                    raise ValueError(f"Label '{label}' without command and is not a standalone label")
                labels_in_order.append((label, rest))
                labeled = 2
        elif labeled:
            if labeled == 1:
                lst = reserved
            else:
                lst = labels_in_order
            lst[-1][1].append(true_datum[1:].strip())  # Remove . and whitespace
        else:
            raise ValueError(f"Error: line {i} '{datum}', all relational labels need a starting label above them")

    # Prepass on the commands to extract and make right any new ones aswell as throw if there is an unknown command
    highest_data: int = 1  # to know how much to allocate (0 is used for some div, mul, .. variations and 1 is used for
    for label_set in (reserved, labels_in_order):  # call and ret)
        for _, rel_labels in label_set:
            for i, rel_label in enumerate(rel_labels.copy()):  # We will only modify idx's
                instruction, data, *_ = rel_label.split(" ", maxsplit=1) + [""]
                instruction = instruction.lower().strip()
                valid_list = valid_commands.get(instruction, None)
                if instruction.isalpha() and len(instruction) == 1:  # Enable single char data
                    rel_labels[i] = str(ord(instruction))  # Has to be str
                    continue
                elif data[1:].isalpha() and len(data[1:]) == 1 and data[0] == "#":
                    data = "#" + str(ord(data[1:]))  # Has to be str
                elif instruction.isnumeric():
                    continue
                elif valid_list is None:
                    raise ValueError(f"Instruction {instruction} is not in the valid command set")
                elif data == [] and valid_list == ():
                    continue
                data1, maybe_data2, *_ = data.split(",", maxsplit=1) + [""]
                data1 = data1.strip()
                maybe_data2 = maybe_data2.strip()
                if 0:
                    print(data1, ",", maybe_data2)
                    print("VL", valid_list)
                match: bool = False
                for valid_pattern in valid_list:
                    part1, maybe_part2, *_ = valid_pattern.split(" ", maxsplit=1) + [""]
                    part1_start, part1_end, *_ = part1.split("{arg}") + [""]
                    data1data = data1.removeprefix(part1_start).removesuffix(part1_end)
                    if data1 and data1[0] == "d" and data1[1:].isnumeric():
                        num = int(data1[1:])
                        if num > highest_data:
                            highest_data = num
                    if part1 == "" and data1 == "":
                        ...
                    elif not (data1.startswith(part1_start) and data1.endswith(part1_end)  # Can't check here cause labels
                            and data1data):  # .isnumeric()):
                        if 0:
                            print(f"{data1} not matching {part1}")
                        continue
                    part2_start, part2_end, *_ = maybe_part2.split("{aarg}") + [""]
                    data2data = maybe_data2.removeprefix(part2_start).removesuffix(part2_end)
                    if maybe_data2 and maybe_data2[0] == "d" and maybe_data2[1:].isnumeric():
                        num = int(maybe_data2[1:])
                        if num > highest_data:
                            highest_data = num
                    if maybe_part2 == "" and maybe_data2 == "":
                        ...
                    elif not (maybe_data2.startswith(part2_start) and maybe_data2.endswith(part2_end)
                              and data2data):  # .isnumeric()):
                        if 0:
                            print(f"{maybe_data2} not matching {maybe_part2}")
                        continue
                    match = True
                    if isinstance(valid_pattern, c):
                        if 0:
                            print("Is custom")
                        if valid_pattern == "{arg} #{aarg}" and instruction == "sta":
                            rel_labels[i] = [f"lda #{data2data}", f"sta {data1data}"]
                        elif valid_pattern == "({arg}) #{aarg}":
                            rel_labels[i] = [f"lda #{data2data}", f"sta ({data1data})"]
                        elif valid_pattern == "#{arg} {aarg}":
                            rel_labels[i] = [f"lda #{data1data}", f"{instruction} {data2data}"]
                        elif valid_pattern == "{arg} #{aarg}":
                            rel_labels[i] = [f"lda #{data2data}", "sta d0", f"lda {data1data}", f"{instruction} d0"]
                        elif valid_pattern == "#{arg} #{aarg}":
                            rel_labels[i] = [f"lda #{data2data}", "sta d0", f"lda #{data1data}", f"{instruction} d0"]
                        elif instruction == "call":
                            rel_labels[i] = ["lda #idx", "sta d1", "lda #6", "add d1", "sta d1", f"jmp {data1data}"]
                        elif instruction == "ret":
                            rel_labels[i] = ["jmp (d1)"]
                        else:  # We also can't detect them yet
                            print("We do not know the indices yet so we can't resolve the jumps")
                    else:
                        if 0:
                            print("No modification needed")
                        rel_labels[i] = f"{instruction} {data}"
                if not match:
                    raise ValueError(f"Data {data} did not match any valid pattern of the instruction {instruction}")
    lookup_table: dict[str, int | str] = {"start": 0}
    f = False
    for r in reserved:
        if r[0] == 0:
            f = True
    if not f:
        reserved.append((0, []))
    data_cell_lookup_table: dict[str, int] = {}  # We currently allocate them all in one chunk
    for label, rel_labels in labels_in_order:
        new_address = get_gap_of_size(len(rel_labels) + 1, reserved, max_address)

        for i, rlabel in enumerate(rel_labels.copy()):  # Do not modify a list you're iterating over
            if isinstance(rlabel, list):
                continue
            inst, data, *_ = rlabel.split(" ", maxsplit=1) + [""]
            if inst in ("jmp", "jnz", "jze", "jle") and data.lstrip('+-').isnumeric():
                reljump = int(data)
                position = i
                last_position = position + reljump
                while position != last_position:
                    if reljump > 0:
                        position += 1
                        if position >= len(rel_labels):
                            raise IndexError("Traversal went out of bounds")
                        if isinstance(rel_labels[position], list):
                            reljump -= 1
                            reljump += len(rel_labels[position])
                    elif reljump < 0:
                        position -= 1
                        if position < 0:
                            raise IndexError("Traversal went out of bounds")
                        if isinstance(rel_labels[position], list):
                            reljump += 1
                            reljump -= len(rel_labels[position])
                rel_labels[i] = f"{inst} {reljump}"

        rel_labels = unnest_iterable(rel_labels)  # Means flattening the list

        if new_address == -1:
            raise ValueError(f"Ran out of available address space (0-{max_address})")

        if label in lookup_table:
            new_address = lookup_table[label]
            for r in reserved:
                if r[0] == new_address:
                    r[1].extend(rel_labels)
        else:
            lookup_table[label] = new_address  # So we know where to jump to
            reserved.append((new_address, rel_labels))
    del labels_in_order  # Unneeded
    data_address = get_gap_of_size(highest_data + 1, reserved, max_address)
    if data_address == -1:
        raise ValueError(f"Ran out of available address space (0-{max_address}) while allocating {highest_data + 1} data cells")
    reserved.append((data_address, ["0"] * (highest_data + 1)))  # Data cells start at 0
    reserved.sort(key=lambda x: x[0])
    # print(reserved)
    cool_reserved = map_to_indexes(max_address, reserved)
    del reserved  # Isn't needed anymore

    for i in range(data_address, data_address + highest_data + 1):
        lookup_table[f"d{i - data_address}"] = i
        lookup_table[f"(d{i - data_address})"] = f"({i})"

    all_placeholders = list(lookup_table.keys()) + ["#idx"]
    i = 0
    for idx, cell in enumerate(cool_reserved.copy()):
        i = idx
        if cell is None:
            cool_reserved[idx] = ""
            continue
        inst, dat, *_ = cell.split(" ") + [""]
        if dat == "":
            cool_reserved[idx] = f"{i} {cell}\n"
            i += 1
            continue
        lookup_table["#idx"] = f"#{idx}"
        dat = replace_placeholders_with_whitespace_condition(dat, all_placeholders, lookup_table)
        if inst in ("jmp", "jnz", "jze", "jle"):
            if dat.startswith("-") or dat.startswith("+"):
                cool_reserved[idx] = f"{i} {inst} {i + int(dat)}\n"
                i += 1
                continue
        cool_reserved[idx] = f"{i} {inst} {dat}\n"
        i += 1
    to_write = ''.join(cool_reserved)
    try:
        fd = os.open(output, os.O_RDWR | os.O_CREAT)
        os.ftruncate(fd, 0)
        fd_length = len(to_write)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, fd_length)
        os.write(fd, to_write.encode("utf-8"))
        os.lseek(fd, 0, 0)  # Go to beginning of file so locking can happen successfully
    except OSError as e:
        print(f"OSError while writing output file: {e}")
        exit(-1)
    finally:
        if fd != 0:
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, fd_length)
            except Exception as e:
                print(f"Error unlocking the file: {e}")
            os.close(fd)


if __name__ == "__main__":
    length = len(sys.argv)
    if length == 1:
        raise ValueError("You need to specify an input, you can specify an output.")
    if length == 2:
        input = sys.argv[1]
        output = "out.asm"
    else:
        input = sys.argv[1]
        output = ""
        for arg in sys.argv[2:]:
            if arg.startswith("-o="):
                output = arg.removeprefix("-o=")
        if not output:
            output = sys.argv[2]
    input = os.path.abspath(input)
    output = os.path.abspath(output)
    if not input.endswith(".rasm") or not os.path.exists(input):
        raise ValueError(f"The input file ({input}) needs to be of type RASM and exist")
    elif not output.endswith(".asm"):
        raise ValueError(f"The output file ({output}) needs to be of type ASM")
    print(f"Reading {input}, writing {output}")
    main(input, output)
