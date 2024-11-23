"""
REG-COMP 2.0
Interface:
py regcomp2.py {input} {output} --target={sspASM/spASM/pASM/x86-64} --logging-mode={DEBUG/INFO/WARN/ERROR}
"""
from collections import defaultdict, deque
from traceback import format_exc
import argparse
import logging
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


class C(str):
    """Custom instruction that needs further processing."""
    def __repr__(self) -> str:
        return "C" + super().__repr__()


class Instruction:
    """
    Represents an instruction in one of these formats:
    - '{inst} {op1}, {op2}'
    - '{inst} {op1}'
    - '{inst}'

    Labels should be resolved before parsing the instruction.
    """
    def __init__(self, raw_instruction: str) -> None:
        self.op_code: str = ""
        self.op1: str = ""
        self.op2: str = ""
        self.raw_d1: str | None = None
        self.raw_d2: str | None = None
        self.process_raw_inst(raw_instruction.strip())
        if self.op_code == "":
            raise ValueError(raw_instruction)

    def process_raw_inst(self, raw_inst: str) -> None:
        """Sets internal values"""
        self.op_code, data, *_ = raw_inst.split(" ", maxsplit=1) + [""]
        self.op1, self.op2, *_ = [d.strip() for d in data.split(",", maxsplit=1) + [""]]
        self.op_code = self.op_code.lower().strip()

    def has_op1(self) -> bool:
        return self.op1 != ""

    def has_op2(self) -> bool:
        return self.op2 != ""

    def is_of_pattern(self, pattern: str | C) -> bool:
        """
        Matches a pattern of the pattern '...{arg}... ...{aarg}...'
        If the pattern matches, it automatically sets self.raw_d1 and self.raw_d2
        """
        part1, maybe_part2, *_ = pattern.split(" ", maxsplit=1) + [""]
        part1_start, part1_end, *_ = part1.split("{arg}") + [""]
        part2_start, part2_end, *_ = maybe_part2.split("{aarg}") + [""]

        raw_data_1: str = self.op1.removeprefix(part1_start).removesuffix(part1_end)
        matched_1: bool = self.op1.startswith(part1_start) and self.op1.endswith(part1_end) and raw_data_1 != "" and "#" not in raw_data_1

        if maybe_part2 != "":
            raw_data_2: str = self.op2.removeprefix(part2_start).removesuffix(part2_end)
            matched_2: bool = self.op2.startswith(part2_start) and self.op2.endswith(part2_end) and raw_data_2 != "" and "#" not in raw_data_2
        elif self.op2:  # To catch the case that part2 is undefined ("") but op2 is defined
            raw_data_2: str = ""
            matched_2: bool = False
        else:
            raw_data_2: str = ""
            matched_2: bool = True

        actually_matched: bool = (matched_1 and matched_2) or (part1 == "" and raw_data_1 == "" and maybe_part2 == "" and raw_data_2 == "")
        if actually_matched:
            self.raw_d1, self.raw_d2 = raw_data_1, raw_data_2
        return actually_matched

    def __str__(self) -> str:
        operands = ", ".join(filter(bool, [self.op1, self.op2]))
        return " ".join(filter(bool, [self.op_code, operands]))

    def __bool__(self) -> bool:
        return self.op_code != ""

    def __repr__(self) -> str:
        repr_str = f"i({self.op_code}"
        if self.has_op1():
            repr_str += f" {self.op1}"
        if self.has_op2():
            repr_str += f", {self.op2}"
        return repr_str + ")"


def get_gap_of_size(t: int, in_this: dict[int, list], until_address: int) -> int:
    # Step 1: Prepare the list of used address ranges
    used_ranges = []

    # Gather the used ranges from the input
    for start, labels in in_this.items():
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


def replace_placeholders_with_non_alpha_conditions(dat, all_placeholders, lookup_table):
    for placeholder in all_placeholders:
        # Create a pattern that ensures no alphabetic characters are adjacent to the placeholder
        pattern = r'(?<![A-Za-z])' + re.escape(placeholder) + r'(?![A-Za-z])'

        # Replace the matched placeholder with the corresponding value from the lookup_table
        dat = re.sub(pattern,
                     lambda m: str(lookup_table.get(m.group(0), m.group(0))),
                     dat)

    return dat


def main(input: str, output: str, target: _ty.Literal["pASM", "x86-64"]) -> None:
    # Works like this:
    # aarg means another arg and is for extra args like
    # sta 10, #2
    # # means literal, it can accept minus and ASCII
    # Using marg instead of arg as the placeholder means it can also take minus.
    # This is reserved for the jumps
    # c() is defined to let the compiler know "I need to change this", let me execute this piece of code for it.
    # This means it's really easy to add your own custom commands or add to existing ones.
    valid_commands = {
        "lda": ("#{arg}", "{arg}", "({arg})"),
        "sta": ("{arg}", "({arg})", C("{arg} #{aarg}"), C("({arg}) #{aarg}"), C("{arg} {aarg}"), C("({arg}) {aarg}")),
        "add": (C("#{arg} {aarg}"), "{arg}", C("{arg} #{aarg}"), C("#{arg} #{aarg}")),
        "sub": (C("#{arg} {aarg}"), "{arg}", C("{arg} #{aarg}"), C("#{arg} #{aarg}")),
        "mul": (C("#{arg} {aarg}"), "{arg}", C("{arg} #{aarg}"), C("#{arg} #{aarg}")),
        "div": (C("#{arg} {aarg}"), "{arg}", C("{arg} #{aarg}"), C("#{arg} #{aarg}")),  # All these c("#{}") just need an "lda #{}" inserted before them
        "jmp": ("{arg}", "({arg})", C("{marg}"), C("({marg})")),  # All the marg just need to be calculated
        "jnz": ("{arg}", "({arg})", C("{marg}"), C("({marg})")),  # into actual addresses and inserted.
        "jze": ("{arg}", "({arg})", C("{marg}"), C("({marg})")),
        "jle": ("{arg}", "({arg})", C("{marg}"), C("({marg})")),
        "stp": ("",),
        "call": (C("{arg}"),),
        "ret": (C(""),),
    }
    # "x86-64": sys.maxsize - 1
    max_address = {"sspASM": 49, "spASM": 99,
                   "pASM": 199, "x86-64": 99999}[target]  # So we don't have an invalid address in the output
    logging.debug(f"Set max_address to {max_address} for target {target}")
    fd = fd_length = 0  # So we know if getting the fd failed
    data: list[str] = []  # So the type checker is happy
    try:
        fd = os.open(input, os.O_RDWR | os.O_CREAT)
        logging.debug(f"Got fd '{fd}' for reading the input file")
        fd_length = os.fstat(fd).st_size
        logging.debug(f"Size of fd '{fd}' is {fd_length}")
        msvcrt.locking(fd, msvcrt.LK_RLCK, fd_length)  # Tries 10 times to lock
        logging.debug(f"Locking for fd '{fd}' proceeded successfully")
        data = os.read(fd, fd_length).decode("utf-8").split("\n")
        logging.debug(f"Reading of fd '{fd}' proceeded successfully, first 10 lines: {data[:10]}")
        os.lseek(fd, 0, 0)  # Go to the beginning of the file so the unlocking can happen successfully
    except OSError as e:
        logging.error(f"OSError while reading input file: {e}")
        sys.exit(-1)
    finally:
        if fd != 0:
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, fd_length)
                logging.debug(f"Unlocking for fd '{fd}' proceeded successfully")
            except Exception as e:
                print(f"Error unlocking the file: {e}")
            os.close(fd)
            logging.debug(f"Fd '{fd}' successfully closed")
        else:
            logging.error("Getting the fd failed.")
            sys.exit(1)

    # Here we iterate once over the raw data, we assign all instructions to a label and convert them into Instructions()
    indices: list[tuple[int, list[Instruction | list[Instruction]]]] = []  # For indices
    labels: defaultdict[str, list[Instruction]] = defaultdict(list)  # For labels without indices
    labeled: int = 0  # 0 = false, 1 = reserved, 2 = labels
    for i, datum in enumerate(data):
        true_datum, comment, *_ = datum.split(";", maxsplit=1) + [""]
        true_datum = true_datum.strip()
        if not datum or not true_datum:
            continue  # Empty line, line with comment
        elif not true_datum.startswith("."):  # Is not a relational label
            label, rest, *_ = true_datum.split(" ", maxsplit=1) + [""]
            if label.isnumeric():
                indices.append((int(label), [Instruction(rest)]))
                labeled = 1
            elif label[:-2].isnumeric() and label[-2:] == "ff":
                last_reserved: tuple[int, list[Instruction]] = indices[-1]
                if last_reserved[0] + len(last_reserved[1]) == int(label[:-2]):
                    last_reserved[1].append(Instruction(rest))
                else:
                    raise ValueError(f"FF-Label '{label}' is not ff, it has to be '{last_reserved[0] + len(last_reserved[1])}ff'")
            else:
                labeled = 2
                if label.endswith(":"):
                    label = label[:-1]
                    if not label in labels:
                        labels[label] = []
                    continue
                elif len(rest) == 0:
                    raise ValueError(f"Label '{label}' without command and is not a standalone label")
                labels[label].append(Instruction(rest))
        elif labeled:  # Is relational with a label to relate to
            true_datum = true_datum[1:].strip()  # Remove . and whitespace
            if labeled == 1:
                indices[-1][1].append(Instruction(true_datum))
            else:  # labeled == 2
                labels[list(labels.keys())[-1]].append(Instruction(true_datum))
        else:  # Is relational with no label to relate to (very rare)
            raise ValueError(f"Error: line {i} '{datum}', all relational labels need a starting label above them")

    logging.debug(f"Indices: {indices}")
    logging.debug(f"Labels: {labels}")
    # All used data labels are stored in this set. s0 and s1 are used for instructions. d{num} is used by the programmer
    data_labels: set[str] = {"s0", "s1"}
    continued_data_labels: list[tuple[str, int]] = []
    labels: list[tuple[str, list[Instruction | list[Instruction]]]] = [(k, v) for k, v in labels.items()]

    # Next we go over all instructions again and check what kind they are, if they are C() we need to change them.
    # We can also use this to throw an error if there is an unrecognized command.
    for label_set in (indices, labels):
        for _, rel_instructions in label_set:
            for i, rel_inst in enumerate(rel_instructions.copy()):  # We will only modify idx's
                valid_list = valid_commands.get(rel_inst.op_code, None)

                # Handling of special cases, like #a, a, 10
                if rel_inst.op_code.isalpha() and len(rel_inst.op_code) == 1:
                    rel_inst.op_code = str(ord(rel_inst.op_code))
                    continue
                elif rel_inst.op1.startswith("#") and rel_inst.op1[1:].isalpha() and len(rel_inst.op1[1:]) == 1:
                    rel_inst.op1 = "#" + str(ord(rel_inst.op1[1:]))
                elif rel_inst.op2.startswith("#") and rel_inst.op2[1:].isalpha() and len(rel_inst.op2[1:]) == 1:
                    rel_inst.op2 = "#" + str(ord(rel_inst.op2[1:]))
                elif rel_inst.op2.count(" ") > 0 or (len(rel_inst.op2[rel_inst.op2.find("#"):]) > 1 and rel_inst.op2[rel_inst.op2.find("#"):].isalpha()):
                    new_instructions = []
                    parts = rel_inst.op2.split(" ")

                    amount_dests = len(rel_inst.op2.replace(" ", "").replace("#", ""))
                    if rel_inst.op1[0].isalpha():
                        optional_prepend = rel_inst.op1[0]
                        start = int(rel_inst.op1[1:])
                        continued_data_labels.append((rel_inst.op1, amount_dests))
                    elif rel_inst.op1.isnumeric():
                        optional_prepend = ''
                        start = int(rel_inst.op1)
                    else:
                        raise ValueError(f"Destination '{rel_inst.op1}' is invalid")
                    destinations = iter([f"{optional_prepend}{i}" for i in range(start, start + amount_dests)])

                    for idx, part in enumerate(parts):
                        if not part.startswith("#"):
                            raise ValueError(f"Continued storing can only be used with literal values, not with '{part}'")
                        placeholder = part[1:]
                        if placeholder.isnumeric():
                            new_instructions.extend([f"lda #{placeholder}", f"sta {next(destinations)}"])
                        else:
                            for char in placeholder:
                                new_instructions.extend([f"lda #{ord(char)}", f"sta {next(destinations)}"])
                    rel_instructions[i] = [Instruction(x) for x in new_instructions]
                    continue
                elif rel_inst.op_code.isnumeric():
                    continue
                elif valid_list is None:
                    raise ValueError(f"Op-Code {rel_inst.op_code} is not in the valid command set")
                logging.debug(f"ValidList for {rel_inst}: {valid_list}")
                matches: list[str | C] = []

                for valid_pattern in valid_list:
                    pattern_matched = rel_inst.is_of_pattern(valid_pattern)

                    if pattern_matched:
                        matches.append(valid_pattern)
                    else:
                        logging.debug(f"{rel_inst.op1} not matching '{'part1'}'")
                        logging.debug(f"{rel_inst.op2} not matching '{'maybe_part2'}'")

                if len(matches) == 0:
                    raise ValueError(f"Operants '{rel_inst.op1}, {rel_inst.op2}' "
                                     f"did not match any valid pattern of the instruction {rel_inst.op_code}")
                else:
                    valid_pattern = max(matches, key=lambda x: len(x))  # The longer the pattern the better the match
                    if rel_inst.raw_d1 and rel_inst.raw_d1[0].isalpha() and rel_inst.raw_d1[1:].isnumeric():
                        data_labels.add(rel_inst.raw_d1)
                    elif rel_inst.raw_d2 and rel_inst.raw_d2[0].isalpha() and rel_inst.raw_d2[1:].isnumeric():
                        data_labels.add(rel_inst.raw_d2)
                    if isinstance(valid_pattern, C):
                        logging.debug(f"Instruction '{rel_inst}' is custom with pattern '{valid_pattern}'")
                        if valid_pattern == "{arg} #{aarg}" and rel_inst.op_code == "sta":
                            rel_instructions[i] = [f"lda #{rel_inst.raw_d2}",
                                                   f"sta {rel_inst.raw_d1}"]
                        elif valid_pattern == "({arg}) #{aarg}" and rel_inst.op_code == "sta":
                            rel_instructions[i] = [f"lda #{rel_inst.raw_d2}",
                                                   f"sta ({rel_inst.raw_d1})"]
                        elif valid_pattern == "#{arg} {aarg}":
                            rel_instructions[i] = [f"lda #{rel_inst.raw_d1}",
                                                   f"{rel_inst.op_code} {rel_inst.raw_d2}"]
                        elif valid_pattern == "{arg} #{aarg}":
                            rel_instructions[i] = [f"lda #{rel_inst.raw_d2}",
                                                   "sta s0",
                                                   f"lda {rel_inst.raw_d1}",
                                                   f"{rel_inst.op_code} s0"]
                        elif valid_pattern == "#{arg} #{aarg}":
                            rel_instructions[i] = [f"lda #{rel_inst.raw_d2}",
                                                   "sta s0",
                                                   f"lda #{rel_inst.raw_d1}",
                                                   f"{rel_inst.op_code} s0"]
                        elif rel_inst.op_code == "call":
                            rel_instructions[i] = ["lda #idx",
                                                   "sta s1",
                                                   "lda #6",
                                                   "add s1",
                                                   "sta s1",
                                                   f"jmp {rel_inst.raw_d1}"]
                        elif rel_inst.op_code == "ret":
                            rel_instructions[i] = ["jmp (s1)"]
                        elif valid_pattern == "{arg} {aarg}" and rel_inst.op_code == "sta":
                            rel_instructions[i] = [f"lda {rel_inst.raw_d2}",
                                                   f"sta {rel_inst.raw_d1}"]
                        elif valid_pattern == "({arg}) {aarg}" and rel_inst.op_code == "sta":
                            rel_instructions[i] = [f"lda {rel_inst.raw_d2}",
                                                   f"sta ({rel_inst.raw_d1})"]
                        else:  # We also can't detect them yet
                            print("We do not know the indices yet so we can't resolve the jumps")
                        curr_inst = rel_instructions[i]
                        if isinstance(curr_inst, list):  # We do it collectively if a change was made
                            rel_instructions[i] = [Instruction(inst) for inst in curr_inst]
                    else:
                        logging.debug(f"Instructions '{rel_inst}' does not need modification")
            for i, rel_inst in enumerate(rel_instructions.copy()):  # Do not modify a list you're iterating over
                # In the previous iteration we finished modifying the amount of instructions.
                # That means all relational jumps may have had more commands inserted before them.
                # Inserted commands are in a list so we easily know if we need to increment the relational jump or not.
                if isinstance(rel_inst, list):
                    continue
                if ((rel_inst.op_code in ("jmp", "jnz", "jze", "jle")
                        and (rel_inst.op1.startswith("-") or rel_inst.op1.startswith("+")))
                        and rel_inst.op1.lstrip("+-").isnumeric()):
                    reljump = int(rel_inst.op1)
                    position = i
                    last_position = position + reljump
                    while position != last_position:
                        if reljump > 0:
                            position += 1
                            if position >= len(rel_instructions):
                                raise IndexError("Relational Jump: traversal went out of bounds")
                            if isinstance(rel_instructions[position], list):
                                reljump -= 1
                                reljump += len(rel_instructions[position])
                        elif reljump < 0:
                            position -= 1
                            if position < 0:
                                raise IndexError("Relational Jump: traversal went out of bounds")
                            if isinstance(rel_instructions[position], list):
                                reljump += 1
                                reljump -= len(rel_instructions[position])
                    rel_inst.op1 = ("+" if reljump > 0 else "") + str(reljump)
    logging.debug(f"Indices after pre-pass: {indices}")
    logging.debug(f"Labels after pre-pass: {labels}")

    label_lookup_table: dict[str, int | str] = {"start": 0}
    if not any([res[0] == 0 for res in indices]):  # Ensure a start-idx is present
        indices.append((0, []))

    # Here we prefill the reserved list with all index labels as they already know their positions
    # This needs to be done so that we can efficiently calculate new addresses for the labels
    reserved: dict[int, list[Instruction]] = {}
    for label, rel_instructions in indices:
        rel_instructions = unnest_iterable(rel_instructions,
                                           max_depth=1)  # Means flattening the lists in the list; 2d->1d list
        reserved[label] = rel_instructions
    del indices  # Isn't needed anymore
    # Here we allocate the space for all labels and add them to the reserved dict. Sadly we can't resolve everything
    # here yet as all labels need to be allocated for that to happen.
    for label, rel_instructions in labels:  # Here we reserve address space for a label without index
        new_address = get_gap_of_size(len(rel_instructions) + 1, reserved, max_address)

        if new_address == -1:  # If it returns -1 it couldn't find enough space.
            raise ValueError(f"Ran out of available address space (0-{max_address})")

        rel_instructions = unnest_iterable(rel_instructions,
                                           max_depth=1)  # Means flattening the lists in the list; 2d->1d list
        if label in label_lookup_table:
            new_address = label_lookup_table[label]
            reserved[new_address].extend(rel_instructions)
        else:
            label_lookup_table[label] = new_address  # So we know where to jump to
            reserved[new_address] = rel_instructions
    del labels  # Unneeded
    logging.debug(f"Reserved address spaces: {reserved}")

    # Here we allocate all the data cells. Data-cell lookup for all d{num} and s{num},
    # this way we can also allocate them in smaller chunks
    data_cell_lookup_table: dict[str, int] = {}

    # Allocated continued data labels.
    for start_label, length in continued_data_labels:
        identifier, start = start_label[0], int(start_label[1:])
        new_address = get_gap_of_size(length, reserved, max_address)

        if new_address == -1:
            raise ValueError(f"Ran out of available address space (0-{max_address}) "
                             f"while allocating {length} continued data cells")

        for i, label in enumerate(f"{identifier}{i}" for i in range(start, start + length)):
            data_cell_lookup_table[label] = new_address + i
        reserved[new_address] = [Instruction("0")] * length

    queue = deque([(list(data_labels), 0)])  # Queue holds (labels_to_allocate, attempt_level)

    while queue:
        current_labels, level = queue.popleft()
        if not current_labels:  # No labels to allocate in this block
            continue

        size = len(current_labels)
        data_address = get_gap_of_size(size, reserved, max_address)
        if data_address != -1:
            for i, label in enumerate(current_labels):
                data_cell_lookup_table[label] = data_address + i
            reserved[data_address] = [Instruction("0")] * size
        else:  # Allocation failed; split the block into two halves and retry
            if size == 1:
                raise ValueError(f"Ran out of available address space (0-{max_address}) "
                                 f"while allocating {size} data cells")
            logging.warning(f"Ran out of available address space (0-{max_address}) while allocating {size} data cells")
            mid = len(current_labels) // 2
            queue.append((current_labels[:mid], level + 1))  # First half
            queue.append((current_labels[mid:], level + 1))  # Second half

    # Initialize the array with max_address+1 elements set to None
    instruction_list: list[Instruction | None] = [None] * (max_address + 1)
    for start, insts in reserved.items():
        for idx, instructions in enumerate(insts):
            if start + idx <= max_address:  # Ensure we don't exceed the max address
                instruction_list[start + idx] = instructions

    logging.debug(f"File list: {instruction_list}")
    logging.debug(f"Label lookup {label_lookup_table}")
    logging.debug(f"Data cell lookup {data_cell_lookup_table}")

    lookup_table: dict[str, str | int] = {**label_lookup_table, **data_cell_lookup_table}

    # Decommission this sometime soon
    for label, address in data_cell_lookup_table.items():
        lookup_table[label] = address

    all_placeholders = list(lookup_table.keys()) + ["#idx", "#n"]

    n = 0  # Do keeps track of the amount of actual instructions processed
    file_list: list[str] = []
    for idx, cell in enumerate(instruction_list):
        if cell is None:
            continue
        elif cell.op1 == "" and cell.op2 == "":
            file_list.append(f"{idx} {cell}\n")
            n += 1
            continue

        lookup_table["#idx"] = f"#{idx}"
        lookup_table["#n"] = f"#{n}"
        cell.op1 = replace_placeholders_with_non_alpha_conditions(cell.op1, all_placeholders, lookup_table)
        cell.op2 = replace_placeholders_with_non_alpha_conditions(cell.op2, all_placeholders, lookup_table)

        if cell.op_code in ("jmp", "jnz", "jze", "jle"):
            if cell.op1.startswith("-") or cell.op1.startswith("+"):
                file_list.append(f"{idx} {cell.op_code} {idx + int(cell.op1)}\n")
                n += 1
                continue
        file_list.append(f"{idx} {cell}\n")
        n += 1
    logging.debug(f"File list: {file_list}")

    if target in ("sspASM", "spASM", "pASM"):
        to_write = ''.join(file_list).encode("utf-8")
        to_write = to_write.replace(b'\n', b'\r\n')  # For Windows compatibility
    elif target == "x86-64":
        raise ValueError(f"The target {target} is not supported yet")
    else:
        raise ValueError(f"The target '{target}' is not a valid target")

    # Lastly, we write the finished compilation to the output
    try:
        fd = os.open(output, os.O_RDWR | os.O_CREAT | os.O_BINARY)
        fd_length = max(len(to_write), os.fstat(fd).st_size)  # Get bigger of the two, so we always lock everything
        msvcrt.locking(fd, msvcrt.LK_LOCK, fd_length)  # Tries 10 times to lock
        os.ftruncate(fd, 0)  # Needs locking too
        os.write(fd, to_write)
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
    parser = argparse.ArgumentParser(description="REG-COMP: A simple register compiler.")
    parser.add_argument("input", help="Path to the input file")
    parser.add_argument("output", nargs="?", default="out.asm", help="Path to the output file")
    parser.add_argument("-o", nargs="?", default="", help="Path to the output file")
    parser.add_argument("--target", choices=["sspASM", "spASM", "pASM", "x86-64"], required=True,
                        help="Compilation target: pASM or x86-64")
    parser.add_argument("--logging-mode", choices=["DEBUG", "INFO", "WARN", "ERROR"], default="INFO",
                        help="Logging mode (default: INFO)")

    args = parser.parse_args()

    # Setting up the logger
    target = args.target
    logging_mode = args.logging_mode
    logging_level = getattr(logging, logging_mode.upper(), None)
    if logging_level is None:
        logging.error(f"Invalid logging mode: {logging_mode}")
        sys.exit(1)
    logging.basicConfig(level=logging_level, format="%(levelname)s: %(message)s")
    logging.info(f"Starting compilation for target: {target}")

    input = os.path.abspath(args.input)
    output = os.path.abspath(args.o or args.output)
    if not input.endswith(".rasm") or not os.path.exists(input):
        logging.error(f"The input file ({input}) needs to be of type RASM and exist")
    elif not output.endswith(".asm"):
        logging.error(f"The output file ({output}) needs to be of type ASM")
    logging.info(f"Reading {input}, writing {output}")

    try:
        main(input, output, target)
    except Exception as e:
        logging.error(f"An uncaught exception was thrown in the compilation process: \"{e}\"")
        actual_error = format_exc()
        for line in actual_error.strip().split("\n"):
            logging.error(line)
