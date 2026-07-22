#!/usr/bin/env python3
"""Run a hardcoded batch of MASt3R reconstruction CLI tests."""

import subprocess
import sys
import time
from pathlib import Path

# Backup
# TESTS = [
#     # Edit this list manually. Each src can be a string or a list of strings.
#     # {"test_name": "example_test", "src": ["../data/example_images"]},
#     # {"test_name": "room_1_scen_1", "src": ["../dataset/custom/room_1/images/scen_1"]},
#     {"test_name": "room_1_scen_2", "src": ["../dataset/custom/room_1/images/scen_2"]},
#     {"test_name": "room_1_scen_3", "src": ["../dataset/custom/room_1/images/scen_3"]},
#     {"test_name": "room_1_scen_4", "src": ["../dataset/custom/room_1/images/scen_4"]},
# ]

TESTS_ETH3D = [
    # Edit this list manually. Each src can be a string or a list of strings.
    # {"test_name":custom_ "example_test", "src": ["../data/example_images"]},
    # {"test_name": "eth3d_botanical_garden_2", "src": ["../dataset/eth3d/botanical_garden/images/scen_2"]},
    # {"test_name": "eth3d_botanical_garden_3", "src": ["../dataset/eth3d/botanical_garden/images/scen_3"]},
    # {"test_name": "eth3d_botanical_garden_4", "src": ["../dataset/eth3d/botanical_garden/images/scen_4"]},
    # {"test_name": "eth3d_boulders_scen_2", "src": ["../dataset/eth3d/boulders/images/scen_2"]},
    # {"test_name": "eth3d_boulders_scen_3", "src": ["../dataset/eth3d/boulders/images/scen_3"]},
    # {"test_name": "eth3d_boulders_scen_4", "src": ["../dataset/eth3d/boulders/images/scen_4"]},
    # {"test_name": "eth3d_bridge_scen_2", "src": ["../dataset/eth3d/bridge/images/scen_2"]},
    # {"test_name": "eth3d_bridge_scen_3", "src": ["../dataset/eth3d/bridge/images/scen_3"]},
    # {"test_name": "eth3d_bridge_scen_4", "src": ["../dataset/eth3d/bridge/images/scen_4"]},
    # {"test_name": "eth3d_door_scen_2", "src": ["../dataset/eth3d/door/images/scen_2"]},
    # {"test_name": "eth3d_door_scen_3", "src": ["../dataset/eth3d/door/images/scen_3"]},
    # {"test_name": "eth3d_door_scen_4", "src": ["../dataset/eth3d/door/images/scen_4"]},
    # {"test_name": "eth3d_exhibition_hall_scen_2", "src": ["../dataset/eth3d/exhibition_hall/images/scen_2"]},
    # {"test_name": "eth3d_exhibition_hall_scen_3", "src": ["../dataset/eth3d/exhibition_hall/images/scen_3"]},
    # {"test_name": "eth3d_exhibition_hall_scen_4", "src": ["../dataset/eth3d/exhibition_hall/images/scen_4"]},
    # {"test_name": "eth3d_lecture_room_scen_2", "src": ["../dataset/eth3d/lecture_room/images/scen_2"]},
    # {"test_name": "eth3d_lecture_room_scen_3", "src": ["../dataset/eth3d/lecture_room/images/scen_3"]},
    # {"test_name": "eth3d_lecture_room_scen_4", "src": ["../dataset/eth3d/lecture_room/images/scen_4"]},
    # {"test_name": "eth3d_lounge_scen_2", "src": ["../dataset/eth3d/lounge/images/scen_2"]},
    # {"test_name": "eth3d_lounge_scen_3", "src": ["../dataset/eth3d/lounge/images/scen_3"]},
    # {"test_name": "eth3d_lounge_scen_4", "src": ["../dataset/eth3d/lounge/images/scen_4"]},
    # {"test_name": "eth3d_observatory_scen_2", "src": ["../dataset/eth3d/observatory/images/scen_2"]},
    # {"test_name": "eth3d_observatory_scen_3", "src": ["../dataset/eth3d/observatory/images/scen_3"]},
    # {"test_name": "eth3d_observatory_scen_4", "src": ["../dataset/eth3d/observatory/images/scen_4"]}, // continue from here
    {"test_name": "eth3d_old_computer_scen_2", "src": ["../dataset/eth3d/old_computer/images/scen_2"]},
    {"test_name": "eth3d_old_computer_scen_3", "src": ["../dataset/eth3d/old_computer/images/scen_3"]},
    {"test_name": "eth3d_old_computer_scen_4", "src": ["../dataset/eth3d/old_computer/images/scen_4"]},
    {"test_name": "eth3d_statue_scen_2", "src": ["../dataset/eth3d/statue/images/scen_2"]},
    {"test_name": "eth3d_statue_scen_3", "src": ["../dataset/eth3d/statue/images/scen_3"]},
    {"test_name": "eth3d_statue_scen_4", "src": ["../dataset/eth3d/statue/images/scen_4"]},
    {"test_name": "eth3d_terrace_2_scen_2", "src": ["../dataset/eth3d/terrace_2/images/scen_2"]},
    {"test_name": "eth3d_terrace_2_scen_3", "src": ["../dataset/eth3d/terrace_2/images/scen_3"]},
    {"test_name": "eth3d_terrace_2_scen_4", "src": ["../dataset/eth3d/terrace_2/images/scen_4"]},
]

TESTS_LLFF = [
    # Edit this list manually. Each src can be a string or a list of strings.
    # {"test_name":custom_ "example_test", "src": ["../data/example_images"]},
    {"test_name": "llff_fern_scen_2", "src": ["../dataset/nerf_llff_data/fern/images/scen_2"]},
    {"test_name": "llff_fern_scen_3", "src": ["../dataset/nerf_llff_data/fern/images/scen_3"]},
    {"test_name": "llff_fern_scen_4", "src": ["../dataset/nerf_llff_data/fern/images/scen_4"]},
    {"test_name": "llff_flower_scen_2", "src": ["../dataset/nerf_llff_data/flower/images/scen_2"]},
    {"test_name": "llff_flower_scen_3", "src": ["../dataset/nerf_llff_data/flower/images/scen_3"]},
    {"test_name": "llff_flower_scen_4", "src": ["../dataset/nerf_llff_data/flower/images/scen_4"]},
    {"test_name": "llff_fortress_scen_2", "src": ["../dataset/nerf_llff_data/fortress/images/scen_2"]},
    {"test_name": "llff_fortress_scen_3", "src": ["../dataset/nerf_llff_data/fortress/images/scen_3"]},
    {"test_name": "llff_fortress_scen_4", "src": ["../dataset/nerf_llff_data/fortress/images/scen_4"]},
    {"test_name": "llff_horns_scen_2", "src": ["../dataset/nerf_llff_data/horns/images/scen_2"]},
    {"test_name": "llff_horns_scen_3", "src": ["../dataset/nerf_llff_data/horns/images/scen_3"]},
    {"test_name": "llff_horns_scen_4", "src": ["../dataset/nerf_llff_data/horns/images/scen_4"]},
    {"test_name": "llff_leaves_scen_2", "src": ["../dataset/nerf_llff_data/leaves/images/scen_2"]},
    {"test_name": "llff_leaves_scen_3", "src": ["../dataset/nerf_llff_data/leaves/images/scen_3"]},
    {"test_name": "llff_leaves_scen_4", "src": ["../dataset/nerf_llff_data/leaves/images/scen_4"]},
    {"test_name": "llff_orchids_scen_2", "src": ["../dataset/nerf_llff_data/orchids/images/scen_2"]},
    {"test_name": "llff_orchids_scen_3", "src": ["../dataset/nerf_llff_data/orchids/images/scen_3"]},
    {"test_name": "llff_orchids_scen_4", "src": ["../dataset/nerf_llff_data/orchids/images/scen_4"]},
    {"test_name": "llff_room_scen_2", "src": ["../dataset/nerf_llff_data/room/images/scen_2"]},
    {"test_name": "llff_room_scen_3", "src": ["../dataset/nerf_llff_data/room/images/scen_3"]},
    {"test_name": "llff_room_scen_4", "src": ["../dataset/nerf_llff_data/room/images/scen_4"]},
    {"test_name": "llff_trex_scen_2", "src": ["../dataset/nerf_llff_data/trex/images/scen_2"]},
    {"test_name": "llff_trex_scen_3", "src": ["../dataset/nerf_llff_data/trex/images/scen_3"]},
    {"test_name": "llff_trex_scen_4", "src": ["../dataset/nerf_llff_data/trex/images/scen_4"]},
]

TESTS_CUSTOM = [
    # Edit this list manually. Each src can be a string or a list of strings.
    # {"test_name":custom_ "example_test", "src": ["../data/example_images"]},
    {"test_name": "custom_room_1_scen_2", "src": ["../dataset/custom/room_1/images/scen_2"]},
    {"test_name": "custom_room_1_scen_3", "src": ["../dataset/custom/room_1/images/scen_3"]},
    {"test_name": "custom_room_1_scen_4", "src": ["../dataset/custom/room_1/images/scen_4"]},
    {"test_name": "custom_office_scen_2", "src": ["../dataset/custom/office/images/scen_2"]},
    {"test_name": "custom_office_scen_3", "src": ["../dataset/custom/office/images/scen_3"]},
    {"test_name": "custom_office_scen_4", "src": ["../dataset/custom/office/images/scen_4"]},
    {"test_name": "custom_synagogue_scen_2", "src": ["../dataset/custom/synagogue/images/scen_2"]},
    {"test_name": "custom_synagogue_scen_3", "src": ["../dataset/custom/synagogue/images/scen_3"]},
    {"test_name": "custom_synagogue_scen_4", "src": ["../dataset/custom/synagogue/images/scen_4"]},
    {"test_name": "custom_statue_scen_2", "src": ["../dataset/custom/statue/images/scen_2"]},
    {"test_name": "custom_statue_scen_3", "src": ["../dataset/custom/statue/images/scen_3"]},
    {"test_name": "custom_statue_scen_4", "src": ["../dataset/custom/statue/images/scen_4"]},
    {"test_name": "custom_cafe_scen_2", "src": ["../dataset/custom/cafe/images/scen_2"]},
    {"test_name": "custom_cafe_scen_3", "src": ["../dataset/custom/cafe/images/scen_3"]},
    {"test_name": "custom_cafe_scen_4", "src": ["../dataset/custom/cafe/images/scen_4"]},
    {"test_name": "custom_room_2_scen_2", "src": ["../dataset/custom/room_2/images/scen_2"]},
    {"test_name": "custom_room_2_scen_3", "src": ["../dataset/custom/room_2/images/scen_3"]},
    {"test_name": "custom_room_2_scen_4", "src": ["../dataset/custom/room_2/images/scen_4"]},
    {"test_name": "custom_bush_1_scen_2", "src": ["../dataset/custom/bush_1/images/scen_2"]},
    {"test_name": "custom_bush_1_scen_3", "src": ["../dataset/custom/bush_1/images/scen_3"]},
    {"test_name": "custom_bush_1_scen_4", "src": ["../dataset/custom/bush_1/images/scen_4"]},
    {"test_name": "custom_room_3_scen_2", "src": ["../dataset/custom/room_3/images/scen_2"]},
    {"test_name": "custom_room_3_scen_3", "src": ["../dataset/custom/room_3/images/scen_3"]},
    {"test_name": "custom_room_3_scen_4", "src": ["../dataset/custom/room_3/images/scen_4"]},
    {"test_name": "custom_room_5_scen_2", "src": ["../dataset/custom/room_5/images/scen_2"]},
    {"test_name": "custom_room_5_scen_3", "src": ["../dataset/custom/room_5/images/scen_3"]},
    {"test_name": "custom_room_5_scen_4", "src": ["../dataset/custom/room_5/images/scen_4"]},
    {"test_name": "custom_millenium_falcon_scen_2", "src": ["../dataset/custom/millenium_falcon/images/scen_2"]},
    {"test_name": "custom_millenium_falcon_scen_3", "src": ["../dataset/custom/millenium_falcon/images/scen_3"]},
    {"test_name": "custom_millenium_falcon_scen_4", "src": ["../dataset/custom/millenium_falcon/images/scen_4"]},
    {"test_name": "custom_tennis_table_scen_2", "src": ["../dataset/custom/tennis_table/images/scen_2"]},
    {"test_name": "custom_tennis_table_scen_3", "src": ["../dataset/custom/tennis_table/images/scen_3"]},
    {"test_name": "custom_tennis_table_scen_4", "src": ["../dataset/custom/tennis_table/images/scen_4"]},
    {"test_name": "custom_church_scen_2", "src": ["../dataset/custom/church/images/scen_2"]},
    {"test_name": "custom_church_scen_3", "src": ["../dataset/custom/church/images/scen_3"]},
    {"test_name": "custom_church_scen_4", "src": ["../dataset/custom/church/images/scen_4"]},
    {"test_name": "custom_mural_bird_scen_2", "src": ["../dataset/custom/mural_bird/images/scen_2"]},
    {"test_name": "custom_mural_bird_scen_3", "src": ["../dataset/custom/mural_bird/images/scen_3"]},
    {"test_name": "custom_mural_bird_scen_4", "src": ["../dataset/custom/mural_bird/images/scen_4"]},
    {"test_name": "custom_kettle_scen_2", "src": ["../dataset/custom/kettle/images/scen_2"]},
    {"test_name": "custom_kettle_scen_3", "src": ["../dataset/custom/kettle/images/scen_3"]},
    {"test_name": "custom_kettle_scen_4", "src": ["../dataset/custom/kettle/images/scen_4"]},
    {"test_name": "custom_spqr_scen_2", "src": ["../dataset/custom/spqr/images/scen_2"]},
    {"test_name": "custom_spqr_scen_3", "src": ["../dataset/custom/spqr/images/scen_3"]},
    {"test_name": "custom_spqr_scen_4", "src": ["../dataset/custom/spqr/images/scen_4"]},
    {"test_name": "custom_mural_king_scen_2", "src": ["../dataset/custom/mural_king/images/scen_2"]},
    {"test_name": "custom_mural_king_scen_3", "src": ["../dataset/custom/mural_king/images/scen_3"]},
    {"test_name": "custom_mural_king_scen_4", "src": ["../dataset/custom/mural_king/images/scen_4"]},
    {"test_name": "custom_miscellaneous_scen_2", "src": ["../dataset/custom/miscellaneous/images/scen_2"]},
    {"test_name": "custom_miscellaneous_scen_3", "src": ["../dataset/custom/miscellaneous/images/scen_3"]},
    {"test_name": "custom_miscellaneous_scen_4", "src": ["../dataset/custom/miscellaneous/images/scen_4"]},
    {"test_name": "custom_gate_scen_2", "src": ["../dataset/custom/gate/images/scen_2"]},
    {"test_name": "custom_gate_scen_3", "src": ["../dataset/custom/gate/images/scen_3"]},
    {"test_name": "custom_gate_scen_4", "src": ["../dataset/custom/gate/images/scen_4"]},
    {"test_name": "custom_iron_scen_2", "src": ["../dataset/custom/iron/images/scen_2"]},
    {"test_name": "custom_iron_scen_3", "src": ["../dataset/custom/iron/images/scen_3"]},
    {"test_name": "custom_iron_scen_4", "src": ["../dataset/custom/iron/images/scen_4"]},
    {"test_name": "custom_bush_2_scen_2", "src": ["../dataset/custom/bush_2/images/scen_2"]},
    {"test_name": "custom_bush_2_scen_3", "src": ["../dataset/custom/bush_2/images/scen_3"]},
    {"test_name": "custom_bush_2_scen_4", "src": ["../dataset/custom/bush_2/images/scen_4"]},
    {"test_name": "custom_room_4_scen_2", "src": ["../dataset/custom/room_4/images/scen_2"]},
    {"test_name": "custom_room_4_scen_3", "src": ["../dataset/custom/room_4/images/scen_3"]},
    {"test_name": "custom_room_4_scen_4", "src": ["../dataset/custom/room_4/images/scen_4"]},
]

TESTS = [*TESTS_ETH3D]


# None means: use reconstruct_cli.py default, currently 3 runs.
RUNS_PER_TEST = None
SLEEP_BETWEEN_TESTS_SECONDS = 30
STOP_ON_FAILURE = True


def normalize_src(src):
    if isinstance(src, str):
        return [src]
    return list(src)


def main():
    script_path = Path(__file__).resolve().with_name("reconstruct_cli.py")
    if not TESTS:
        print("No tests configured. Edit TESTS in this file and run again.")
        return

    for index, test in enumerate(TESTS, start=1):
        test_name = test["test_name"]
        src_values = normalize_src(test["src"])
        command = [
            sys.executable,
            script_path.name,
            "--test-name",
            test_name,
            "--src",
            *src_values,
        ]
        runs = test.get("runs", RUNS_PER_TEST)
        if runs is not None:
            command.extend(["--runs", str(runs)])

        print(f"[{index}/{len(TESTS)}] Running {test_name}")
        result = subprocess.run(command, cwd=script_path.parent, check=False)
        if result.returncode != 0:
            print(f"[{index}/{len(TESTS)}] FAILED {test_name}: exit {result.returncode}")
            if STOP_ON_FAILURE:
                raise SystemExit(result.returncode)
        else:
            print(f"[{index}/{len(TESTS)}] Finished {test_name}")

        if index < len(TESTS) and SLEEP_BETWEEN_TESTS_SECONDS > 0:
            print(f"Sleeping {SLEEP_BETWEEN_TESTS_SECONDS}s before next test")
            time.sleep(SLEEP_BETWEEN_TESTS_SECONDS)


if __name__ == "__main__":
    main()
