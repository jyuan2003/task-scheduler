import sys
import subprocess
from pathlib import Path
import time
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from data_generation.make_graphs import make_dataset


def main():
    bs = [1, 8, 128]
    weight = ["low_var", "high_var"]
    structure = ["random", "cluster"]
    nproc = [1, 2, 4, 8, 16]
    init = ["random", "contiguous"]
    density = ["sparse", "medium", "dense"]
    rep = ["raw", "compressed", "sharded", "compressed_sharded"]
    lo = 1
    hi = 100000

    subprocess.run(
        [
            "mpic++",
            "-std=c++20",
            "-w",
            "mpi/mpi_main.cpp",
            "mpi/scheduler_utils.cpp",
            "graph_representation/graph_utils.cpp",
            "-o",
            "mpi/mpi_main",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    for w in weight:
        for d in density:
            for i in init:
                for r in rep:
                    for s in structure:
                        for b in bs:
                            for proc in nproc:
                                l = lo
                                h = hi
                                max_n = 0
                                while l <= h:
                                    n = (l + h) // 2
                                    fpath = make_dataset(["--nodes", str(n), "--density", d, "--weight", w, "--structure", s, "--save_path", "data_generation/raw_dataset/"])
                                    
                                    try:
                                        res = subprocess.run(["mpirun", "-np", str(proc), str(PROJECT_ROOT / "mpi" / "mpi_main"), "-r", r, "-f", str(fpath), "-i", "5", "-b", str(b), "-m", str(i)], cwd=PROJECT_ROOT, check=True, timeout=30, capture_output=True, text=True)
                                        l = n + 1
                                        max_n = n
                                    except:
                                        h = n - 1
                                print(f"$$$$$$$$$$$$${w}_{d}_{i}_{r}_{s}_{b}_{proc}$$$$$$$$$$$$")
                                print("max_n: ", max_n)
                                if (res): print(res.stdout)
                                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4$$")
                            

if __name__ == "__main__":
    main()
