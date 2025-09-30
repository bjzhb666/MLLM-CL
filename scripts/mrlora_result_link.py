import os
import os.path
import sys

__doc__ = f"""\
reformat results/DCL/each_dataset as model_dataset format.
Usage: python {sys.argv[0]} /path/to/DCL
       python {sys.argv[0]} ACL

e.g. python {sys.argv[0]} DCL
It will run `ln -s ../each_dataset/RS/LoRA-task1 DCL/model_dataset/RS_RS`
            `ln -s ../each_dataset/RS/LoRA-task2 DCL/model_dataset/Med_RS`
and creates M*N DCL/model_dataset/*_*
"""


def safe_slink(src: str, dst: str, dry_run: bool = False):
    if dry_run:
        print(f"[DRY RUN] {src} -> {dst}")
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # refresh symlink, os.path.exists does not work for broken symlink
    try:
        os.unlink(dst)
    except Exception:
        pass
    os.symlink(src, dst)


def link_all_subdirs(folder: str):
    assert folder.endswith("ACL") or folder.endswith("DCL"), (
        f"{folder} must end with ACL or DCL"
    )

    if folder.endswith("ACL"):
        categories = ["OCR", "Math", "VP", "APP"]
    if folder.endswith("DCL"):
        categories = ["RS", "Med", "AD", "Sci", "Fin"]

    for d_id, dataset in enumerate(categories):
        for m_id, model in enumerate(categories):
            safe_slink(
                f"../each_dataset/{dataset}/LoRA-task{m_id + 1}",
                f"{folder}/model_dataset/{model}_{dataset}",
                dry_run=False,
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        link_all_subdirs(sys.argv[1])
