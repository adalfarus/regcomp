"""
REG-COMP 2.0
Interface:
py regcomp2.py {input} {output} --target={pASM/pASM.c/x86-64} --target-operant-size={} --target-memory-size={} --logging-mode={DEBUG/INFO/WARN/ERROR}
"""
from collections import defaultdict, deque
from traceback import format_exc
import argparse
import logging
import msvcrt
import sys
import os
import re
import io

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
            logging.error(raw_instruction)
            sys.exit(1)

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


T = _ty.TypeVar('T')


class AsmAddressSpaceList(_ty.Generic[T]):
    """
    Initializes an empty address space list where unset elements return None.
    Only non-None elements are stored internally.
    """
    def __init__(self) -> None:
        self._data: dict[int, T] = {}

    def __getitem__(self, index: int) -> T | None:
        """
        Gets the value at the given index. If the index is unset, returns None.
        """
        if index < 0:
            raise IndexError("AsmAddressSpaceList index out of range")
        return self._data.get(index, None)

    def __setitem__(self, index: int, value: T | None) -> None:
        """
        Sets the value at the given index. If the value is None, the index is removed.
        """
        if index < 0:
            raise IndexError("AsmAddressSpaceList index out of range")
        if value is None:
            self._data.pop(index, None)  # Remove the index if the value is None
        else:
            self._data[index] = value

    def __len__(self) -> int:
        """
        Returns the total size of the address space (including unset indices).
        Since the list is sparse, we return the largest index + 1.
        """
        return max(self._data.keys(), default=-1) + 1

    def __repr__(self) -> str:
        """
        Returns a string representation of the sparse list, showing only set elements.
        """
        return f"<AsmAddressSpaceList size={len(self)} initialized={len(self._data)} data={self._data}>"

    def __iter__(self) -> _ty.Iterator[T | None]:
        """
        Iterates through all indices, returning their values or None for unset indices.
        """
        max_index = len(self)
        for i in range(max_index):
            yield self[i]

    def keys(self) -> _ty.KeysView[T]:
        """
        Returns the set indices of the list (indices with non-None values).
        """
        return self._data.keys()

    def values(self) -> _ty.ValuesView[T]:
        """
        Returns the non-None values stored in the list.
        """
        return self._data.values()

    def items(self) -> _ty.ItemsView[int, T]:
        """
        Returns a dictionary-style view of set indices and their corresponding values.
        """
        return self._data.items()


def main(input: str, output: str, target: _ty.Literal["pASM", "pASM.c", "x86-64"],
         target_memory_size: int = 0, target_operant_size: int = 2) -> None:
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
    # So we don't have an invalid address in the output
    max_memory_size = {"pASM": 199,
                       "pASM.c": 4_294_967_295,
                       "x86-64": sys.maxsize - 1}[target]
    if target_memory_size > max_memory_size:
        logging.error(f"Target memory size {target_memory_size} is larger than the maximum memory size "
                      f"({max_memory_size}) of the chosen target {target}")
        sys.exit(1)
    chosen_memory_size = target_memory_size or max_memory_size  # If 0, default to max
    logging.info(f"Set chosen_memory_size to {chosen_memory_size} "
                 f"for target {target} with max_memory_size of {max_memory_size}")
    fd = fd_length = 0  # So we know if getting the fd failed
    raw_data: bytes = b""  # So the type checker is happy
    skip_custom: bool = input.lower().endswith((".txt", ".pasm", ".p"))
    _expr = "isn't" if skip_custom else "is"
    logging.info(f"Set skip_custom to {skip_custom}, as the input {_expr} of type RASM")
    try:
        fd = os.open(input, os.O_RDWR | os.O_CREAT | os.O_BINARY)
        logging.debug(f"Got fd '{fd}' for reading the input file")
        fd_length = os.fstat(fd).st_size
        logging.debug(f"Size of fd '{fd}' is {fd_length}")
        msvcrt.locking(fd, msvcrt.LK_RLCK, fd_length)  # Tries 10 times to lock
        logging.debug(f"Locking for fd '{fd}' proceeded successfully")
        raw_data = os.read(fd, fd_length)
        logging.debug(f"Reading of fd '{fd}' proceeded successfully, first 10 lines: {raw_data[:10]}")
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

    if input.lower().endswith(".p"):
        logging.info(f"Converting .p to .pasm")
        # Instruction set mapping for reverse translation
        INSTRUCTION_SET = {
            10: "LDA_IMM",  # LDA #xx
            11: "LDA_DIR",  # LDA xx
            12: "LDA_IND",  # LDA (xx)
            20: "STA_DIR",  # STA xx
            21: "STA_IND",  # STA (xx)
            30: "ADD_DIR",  # ADD xx
            40: "SUB_DIR",  # SUB xx
            50: "MUL_DIR",  # MUL xx
            60: "DIV_DIR",  # DIV xx
            70: "JMP_DIR",  # JMP xx
            71: "JMP_IND",  # JMP (xx)
            80: "JNZ_DIR",  # JNZ xx
            81: "JNZ_IND",  # JNZ (xx)
            90: "JZE_DIR",  # JZE xx
            91: "JZE_IND",  # JZE (xx)
            92: "JLE_DIR",  # JLE xx
            93: "JLE_IND",  # JLE (xx)
            99: "STP"  # STP (no operand)
        }
        pasm_lines = []
        with io.BytesIO(raw_data) as f:
            # Read header
            magic = f.read(4).decode("utf-8")
            if magic != "EMUL":
                logging.error(f"Invalid magic number: {magic}")
                sys.exit(1)
            operand_size = int.from_bytes(f.read(1), "little")
            memory_size = int.from_bytes(f.read(4), "little")
            logging.info(f"Header: Magic={magic}, Operand Size={operand_size}, Memory Size={memory_size}")

            if memory_size > chosen_memory_size:
                logging.error(f"Read memory size {memory_size} is larger than chosen memory size {chosen_memory_size}")
                sys.exit(1)
            elif operand_size > target_operant_size:
                logging.error(f"Read operant size {operand_size} is larger than target operant size {target_operant_size}")
                sys.exit(1)
            logging.info(f"Reset chosen memory size to {memory_size} to match read memory size")
            chosen_memory_size = memory_size
            # Initialize output representation
            address = 0
            # Parse instructions and data
            while True:
                # Read opcode
                byte = f.read(1)
                if not byte:
                    break  # End of file reached
                opcode = int.from_bytes(byte, "little")
                if opcode in INSTRUCTION_SET:  # Recognized opcode
                    instruction = INSTRUCTION_SET[opcode]
                    if "STP" == instruction:  # No operand
                        f.read(operand_size)  # Need to discard the operant bits
                        pasm_lines.append(f"{address:02} {instruction}")
                    else:
                        operand_bytes = f.read(operand_size)
                        if len(operand_bytes) < operand_size:
                            logging.error(f"Unexpected end of file when reading operand at address {address:02}")
                            sys.exit(1)
                        operand = int.from_bytes(operand_bytes, "little", signed=True)
                        if instruction.endswith("_IND"):  # Indirect addressing
                            pasm_lines.append(f"{address:02} {instruction.removesuffix('_IND')} ({operand})")
                        elif instruction.endswith("_IMM"):  # Immediate addressing
                            pasm_lines.append(f"{address:02} {instruction.removesuffix('_IMM')} #{operand}")
                        else:  # Direct addressing
                            pasm_lines.append(f"{address:02} {instruction.removesuffix('_DIR')} {operand}")
                else:  # Treat as raw data
                    operand_bytes = f.read(operand_size)
                    if len(operand_bytes) < operand_size:
                        logging.error(f"Unexpected end of file when reading data at address {address:02}")
                        sys.exit(1)
                    # data_value = (opcode << (8 + (operand_size * 8) - 16)) | int.from_bytes(operand_bytes, "little")
                    # data_value = int.from_bytes(data_value.to_bytes(operand_size, "little"), "little", signed=True)
                    data_value = int.from_bytes(operand_bytes, "little", signed=True)
                    # if data_value == 0:  # 0 is either 0 or NOP, but no worries, all regs are 0/NOP by default
                    #     address += 1
                    #     continue
                    pasm_lines.append(f"{address:02} {data_value}")
                address += 1
        data: list[str] = pasm_lines
    else:
        data: list[str] = raw_data.decode("utf-8").split("\n")

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
                    logging.error(f"FF-Label '{label}' is not ff, it has to be '{last_reserved[0] + len(last_reserved[1])}ff'")
                    sys.exit(1)
            else:
                labeled = 2
                if label.endswith(":"):
                    label = label[:-1]
                    if not label in labels:
                        labels[label] = []
                    continue
                elif len(rest) == 0:
                    logging.error(f"Label '{label}' without command and is not a standalone label")
                    sys.exit(1)
                labels[label].append(Instruction(rest))
        elif labeled:  # Is relational with a label to relate to
            true_datum = true_datum[1:].strip()  # Remove . and whitespace
            if labeled == 1:
                indices[-1][1].append(Instruction(true_datum))
            else:  # labeled == 2
                labels[list(labels.keys())[-1]].append(Instruction(true_datum))
        else:  # Is relational with no label to relate to (very rare)
            logging.error(f"Error: line {i} '{datum}', all relational labels need a starting label above them")
            sys.exit(1)

    logging.debug(f"Indices: {indices}")
    logging.debug(f"Labels: {labels}")
    # All used data labels are stored in this set. s0 and s1 are used for instructions. d{num} is used by the programmer
    data_labels: set[str] = {"s0", "s1"}
    continued_data_labels: list[tuple[str, int]] = []
    labels: list[tuple[str, list[Instruction | list[Instruction]]]] = [(k, v) for k, v in labels.items()]

    # Next we go over all instructions again and check what kind they are, if they are C() we need to change them.
    # We can also use this to throw an error if there is an unrecognized command.
    if skip_custom:
        logging.info(f"Skipping custom patterns, for more info enable logging debug mode")
        logging.info(f"Skipping rel-jump adaptations, for more info enable logging debug mode")
    for label_set in (indices, labels):
        for _, rel_instructions in label_set:
            for i, rel_inst in enumerate(rel_instructions.copy()):  # We will only modify idx's
                valid_list = valid_commands.get(rel_inst.op_code, None)

                # Handling of special cases, like #a, a, 10
                if rel_inst.op_code.isalpha() and len(rel_inst.op_code) == 1:
                    rel_inst.op_code = str(ord(rel_inst.op_code))
                    continue
                elif rel_inst.op_code.isnumeric():
                    continue
                elif valid_list is None:
                    logging.error(f"Op-Code {rel_inst.op_code} is not in the valid command set")
                    sys.exit(1)

                if rel_inst.op1.startswith("#") and rel_inst.op1[1:].isalpha() and len(rel_inst.op1[1:]) == 1:
                    rel_inst.op1 = "#" + str(ord(rel_inst.op1[1:]))
                if rel_inst.op2.startswith("#") and rel_inst.op2[1:].isalpha() and len(rel_inst.op2[1:]) == 1:
                    rel_inst.op2 = "#" + str(ord(rel_inst.op2[1:]))
                if rel_inst.op2.count(" ") > 0 or (len(rel_inst.op2[rel_inst.op2.find("#"):]) > 1 and rel_inst.op2[rel_inst.op2.find("#"):].isalpha()):
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
                        logging.error(f"Destination '{rel_inst.op1}' is invalid")
                        sys.exit(1)
                    destinations = iter([f"{optional_prepend}{i}" for i in range(start, start + amount_dests)])

                    for idx, part in enumerate(parts):
                        if not part.startswith("#"):
                            logging.error(f"Continued storing can only be used with literal values, not with '{part}'")
                            sys.exit(1)
                        placeholder = part[1:]
                        if placeholder.isnumeric():
                            new_instructions.extend([f"lda #{placeholder}", f"sta {next(destinations)}"])
                        else:
                            for char in placeholder:
                                new_instructions.extend([f"lda #{ord(char)}", f"sta {next(destinations)}"])
                    rel_instructions[i] = [Instruction(x) for x in new_instructions]
                    continue
                logging.debug(f"ValidList for {rel_inst}: {valid_list}")
                matches: list[str | C] = []

                for valid_pattern in valid_list:
                    if isinstance(valid_pattern, C) and skip_custom:
                        logging.debug(f"Skipped custom pattern {valid_pattern}")
                        continue
                    pattern_matched = rel_inst.is_of_pattern(valid_pattern)

                    if pattern_matched:
                        matches.append(valid_pattern)
                    else:
                        logging.debug(f"{rel_inst.op1} not matching '{'part1'}'")
                        logging.debug(f"{rel_inst.op2} not matching '{'maybe_part2'}'")

                if len(matches) == 0:
                    logging.error(f"Operants '{rel_inst.op1}, {rel_inst.op2}' "
                                  f"did not match any valid pattern of the instruction {rel_inst.op_code}")
                    sys.exit(1)
                else:
                    valid_pattern = max(matches, key=lambda x: len(x))  # The longer the pattern the better the match
                    if rel_inst.raw_d1 and rel_inst.raw_d1[0].isalpha() and rel_inst.raw_d1[1:].isnumeric():
                        data_labels.add(rel_inst.raw_d1)
                    if rel_inst.raw_d2 and rel_inst.raw_d2[0].isalpha() and rel_inst.raw_d2[1:].isnumeric():
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
                if skip_custom:  # No rel jumps in default pasm
                    break
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
    if skip_custom:
        logging.info(f"Skipping all instruction_list flattening")
    for label, rel_instructions in indices:
        if not skip_custom:
            rel_instructions = unnest_iterable(rel_instructions,
                                               max_depth=1)  # Means flattening the lists in the list; 2d->1d list
        reserved[label] = rel_instructions
    del indices  # Isn't needed anymore
    # Here we allocate the space for all labels and add them to the reserved dict. Sadly we can't resolve everything
    # here yet as all labels need to be allocated for that to happen.
    if skip_custom:
        logging.info(f"Skipping label allocation")
    for label, rel_instructions in labels:  # Here we reserve address space for a label without index
        if skip_custom:
            break
        rel_instructions = unnest_iterable(rel_instructions,
                                           max_depth=1)  # Means flattening the lists in the list; 2d->1d list

        new_address = get_gap_of_size(len(rel_instructions) + 1, reserved, chosen_memory_size)

        if new_address == -1:  # If it returns -1 it couldn't find enough space.
            logging.error(f"Ran out of available address space (0-{chosen_memory_size})")
            sys.exit(1)
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
    if not skip_custom:
        for start_label, length in continued_data_labels:
            # The first label is detected at a previous step, so we prevent reallocation of it
            data_labels.remove(start_label)  # when allocating data labels later
            identifier, start = start_label[0], int(start_label[1:])
            new_address = get_gap_of_size(length, reserved, chosen_memory_size)

            if new_address == -1:
                logging.error(f"Ran out of available address space (0-{chosen_memory_size}) "
                              f"while allocating {length} continued data cells")
                sys.exit(1)

            for i, label in enumerate(f"{identifier}{i}" for i in range(start, start + length)):
                data_cell_lookup_table[label] = new_address + i
            reserved[new_address] = [Instruction("0")] * length

        queue = deque([(list(data_labels), 0)])  # Queue holds (labels_to_allocate, attempt_level)

        while queue:
            current_labels, level = queue.popleft()
            if not current_labels:  # No labels to allocate in this block
                continue

            size = len(current_labels)
            data_address = get_gap_of_size(size, reserved, chosen_memory_size)
            if data_address != -1:
                for i, label in enumerate(current_labels):
                    data_cell_lookup_table[label] = data_address + i
                reserved[data_address] = [Instruction("0")] * size
            else:  # Allocation failed; split the block into two halves and retry
                if size == 1:
                    logging.error(f"Ran out of available address space (0-{chosen_memory_size}) "
                                  f"while allocating {size} data cells")
                    sys.exit(1)
                logging.warning(f"Ran out of available address space (0-{chosen_memory_size}) while allocating {size} data cells")
                mid = len(current_labels) // 2
                queue.append((current_labels[:mid], level + 1))  # First half
                queue.append((current_labels[mid:], level + 1))  # Second half
    else:
        logging.info(f"Skipping data label and continued data label allocation and resolution")

    # Initialize the array with max_memory_size+1 elements set to None
    instruction_list: AsmAddressSpaceList[Instruction] = AsmAddressSpaceList()
    for start, insts in reserved.items():
        for idx, instructions in enumerate(insts):
            if start + idx <= chosen_memory_size:  # Ensure we don't exceed the max address
                instruction_list[start + idx] = instructions

    logging.debug(f"File list: {instruction_list}")
    logging.info(f"Label lookup: {', '.join([f'{k}->{v}' for k, v in label_lookup_table.items()])}")  # So you can debug the result
    logging.info(f"Data cell lookup: {', '.join([f'{k}->{v}' for k, v in data_cell_lookup_table.items()])}")  # So you can debug the result

    lookup_table: dict[str, str | int] = {**label_lookup_table, **data_cell_lookup_table}

    # Decommission this sometime soon
    for label, address in data_cell_lookup_table.items():
        lookup_table[label] = address

    all_placeholders = list(lookup_table.keys()) + ["#idx", "#n"]

    n = idx = 0  # Do keeps track of the amount of actual instructions processed
    file_list: list[str] = []
    logging.debug(f"Skipping placeholder replacements and rel-jump resolutions, for more info enable logging debug mode")
    for idx, cell in sorted(instruction_list.items(), key=lambda x: x[0]):
        # if cell is None:
        #     continue
        # elif cell.op1 == "" and cell.op2 == "":
        if cell.op1 == "" and cell.op2 == "":
            file_list.append(f"{idx} {cell}\n")
            n += 1
            continue

        lookup_table["#idx"] = f"#{idx}"
        lookup_table["#n"] = f"#{n}"
        if not skip_custom:
            cell.op1 = replace_placeholders_with_non_alpha_conditions(cell.op1, all_placeholders, lookup_table)
            cell.op2 = replace_placeholders_with_non_alpha_conditions(cell.op2, all_placeholders, lookup_table)

            if cell.op_code in ("jmp", "jnz", "jze", "jle"):
                if cell.op1.startswith("-") or cell.op1.startswith("+"):
                    file_list.append(f"{idx} {cell.op_code} {idx + int(cell.op1)}\n")
                    n += 1
                    continue
        else:
            logging.debug(f"Skipped placeholder replacement and rel-jump resolution")
        if (int("0" + cell.op1.strip("#()")).bit_length() + 7) // 8 > target_operant_size:
            logging.error(f"Op1 of '{cell}' is larger than the target operant size")
            sys.exit(1)
        elif (int("0" + cell.op2.strip("#()")).bit_length() + 7) // 8 > target_operant_size:
            logging.error(f"Op2 of '{cell}' is larger than the target operant size")
            sys.exit(1)
        file_list.append(f"{idx} {cell}\n")
        n += 1
    logging.debug(f"File list: {file_list}")

    if target == "pASM":
        if not input.lower().endswith((".rasm", ".txt", ".p")):
            logging.error(f"The input file ({input}) needs to be of type RASM or TXT for target {target}, "
                          f"PASM files are already in the right format for this target.")
            sys.exit(1)
        elif not output.lower().endswith(".pasm"):
            logging.error(f"The output file ({output}) needs to be of type PASM, "
                          f"this target cannot interpret any other format")
            sys.exit(1)
        to_write = ''.join(file_list).encode("utf-8")
        to_write = to_write.replace(b'\n', b'\r\n')  # For Windows compatibility
    elif target == "pASM.c":
        if not input.lower().endswith((".rasm", ".txt", ".pasm")):
            logging.error(f"The input file ({input}) needs to be of type RASM, TXT or PASM for target {target}, "
                          f"P files are already in the right format for this target.")
            sys.exit(1)
        elif not output.lower().endswith(".p"):
            logging.error(f"The output file ({output}) needs to be of type P (machine code), "
                          f"this target cannot read any other format")
            sys.exit(1)

        # Instruction set mapping with _DIR and _IND variations
        INSTRUCTION_SET = {
            "NOP": 0,       # Cannot be used
            "LDA_IMM": 10,  # LDA #xx
            "LDA_DIR": 11,  # LDA xx
            "LDA_IND": 12,  # LDA (xx)
            "STA_DIR": 20,  # STA xx
            "STA_IND": 21,  # STA (xx)
            "ADD_DIR": 30,  # ADD xx
            "SUB_DIR": 40,  # SUB xx
            "MUL_DIR": 50,  # MUL xx
            "DIV_DIR": 60,  # DIV xx
            "JMP_DIR": 70,  # JMP xx
            "JMP_IND": 71,  # JMP (xx)
            "JNZ_DIR": 80,  # JNZ xx
            "JNZ_IND": 81,  # JNZ (xx)
            "JZE_DIR": 90,  # JZE xx
            "JZE_IND": 91,  # JZE (xx)
            "JLE_DIR": 92,  # JLE xx
            "JLE_IND": 93,  # JLE (xx)
            "STP": 99  # STP (no operand)
        }
        OPERANT_SIZE = target_operant_size  # Operand size (n bytes)
        MEMORY_SIZE = idx  # Memory size (n bytes)

        # Compile to binary machine code
        instructions = []
        for line in file_list:
            try:
                parts = line.strip().split()
                if len(parts) == 2:  # Data or no operand
                    address = int(parts[0])
                    if parts[1].isdigit():  # Data
                        data_value = int(parts[1])
                        instructions.append((address, data_value, None))  # None for operand
                        continue
                    instruction = parts[1].upper()  # Normalize to uppercase
                    if instruction not in INSTRUCTION_SET:
                        raise ValueError(f"Unknown instruction: {instruction}")
                    opcode = INSTRUCTION_SET[instruction]
                    operand = 0
                elif len(parts) == 3:  # With operand
                    address = int(parts[0])
                    instruction = parts[1].upper()  # Normalize to uppercase
                    raw_operand = parts[2]

                    # Detect addressing mode
                    if raw_operand.startswith("#"):
                        operand = int(raw_operand[1:])  # Strip '#' for immediate
                        opcode = INSTRUCTION_SET.get(f"{instruction}_IMM")
                    elif raw_operand.startswith("(") and raw_operand.endswith(")"):
                        operand = int(raw_operand[1:-1])  # Strip parentheses for indirect
                        opcode = INSTRUCTION_SET.get(f"{instruction}_IND")
                    else:
                        operand = int(raw_operand)  # Direct addressing
                        opcode = INSTRUCTION_SET.get(f"{instruction}_DIR")

                    if opcode is None:
                        raise ValueError(f"Invalid addressing mode for instruction: {instruction}")
                else:
                    raise ValueError(f"Malformed instruction: {line}")

                instructions.append((address, opcode, operand))
            except ValueError as e:
                strp_line = line.rstrip("\n")
                logging.error(f"Error parsing line '{strp_line}': {e}")
                sys.exit(1)

        # Sort instructions by address (optional, if required by emulator)
        instructions.sort(key=lambda x: x[0])

        # Create the binary data
        to_write = b""
        # Write header
        to_write += b"EMUL"  # Magic number
        to_write += OPERANT_SIZE.to_bytes(1, "little")
        to_write += MEMORY_SIZE.to_bytes(4, "little")

        # Write instructions and fill gaps with NOPs
        current_address = 0
        for address, opcode, operand in instructions:
            # Fill gaps with NOPs
            while current_address < address:
                # n-byte NOP opcode/data
                to_write += INSTRUCTION_SET["NOP"].to_bytes(1, "little")
                to_write += (0).to_bytes(OPERANT_SIZE, "little")
                current_address += 1
            # Write instruction or data
            if operand is None:  # Data
                opcode: int
                operand: int
                to_write += INSTRUCTION_SET["NOP"].to_bytes(1, "little")
                to_write += opcode.to_bytes(OPERANT_SIZE, "little", signed=True)  # Write data
            else:  # Instruction
                to_write += opcode.to_bytes(1, "little")  # 1-byte opcode
                to_write += operand.to_bytes(OPERANT_SIZE, "little", signed=True)  # Operand with dynamic size
            current_address += 1
    elif target == "x86-64":
        if not input.lower().endswith((".rasm", ".txt", ".pasm", ".p")):
            logging.error(f"The input file ({input}) needs to be of type RASM, TXT or PASM for target {target}")
            sys.exit(1)
        elif not output.lower().endswith(".obj"):
            logging.error(f"The output file ({output}) needs to be of type OBJ (machine code)")
            sys.exit(1)
        raise ValueError(f"The target {target} is not supported yet")
    else:
        logging.error(f"The target '{target}' is not a valid target")
        sys.exit(1)

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


def positive_nonzero_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a positive integer.")
    return ivalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REG-COMP: A simple register compiler.")
    parser.add_argument("input",
                        help="Path to the input file. The file extension specifies the assembly type: "
                             "TXT uses default address resolution for pASM; "
                             "PASM skips address resolution but performs target-specific tasks; "
                             "P converts the machine code to PASM; "
                             "RASM includes all features plus support for extended opcode.")
    parser.add_argument("output", nargs="?", default="",
                        help="Path to the output file. The file extension is checked to ensure industry standards: "
                             "P specifies binary machine code for the pASM.c emulator; "
                             "PASM specifies assembly text for the pASM interpreter; "
                             "OBJ specifies a standard x86-64 obj file for use with e.g. a linker.")
    parser.add_argument("-o", nargs="?", default="", help="Path to the output file")
    parser.add_argument("--target", choices=["pASM", "pASM.c", "x86-64"], required=True,
                        help="Compilation target: pASM, pASM.c or x86-64")
    parser.add_argument("--target-operant-size", type=positive_nonzero_int, default=2,
                        help="A positive integer specifying the target operant size in bytes")
    parser.add_argument("--target-memory-size", type=positive_nonzero_int, default=0,
                        help="A positive integer specifying the target memory size in bytes")
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
    if not os.path.exists(input):
        logging.error(f"The input file ({input}) needs to exist")
        sys.exit(1)
    elif args.target_operant_size > 4:
        logging.error(f"The target operant size ({args.target_operant_size}) is bigger than the maximum allowed of 4")
        sys.exit(1)
    logging.info(f"Reading {input}, writing {output}")

    try:
        main(input, output, target, args.target_memory_size, args.target_operant_size)
    except Exception as e:
        logging.error(f"An uncaught exception was thrown in the compilation process: \"{e}\"")
        actual_error = format_exc()
        for line in actual_error.strip().split("\n"):
            logging.error(line)
