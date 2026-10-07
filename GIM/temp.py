import os

root = r"F:\FusedTec\Data\GIM2021\IGS\RT\2021"

total = 0

for doy in range(91, 182):

    folder = os.path.join(root, f"{doy:03d}")

    if not os.path.exists(folder):
        print(f"DOY {doy:03d}: 目录不存在")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.startswith("irtg") and f.endswith(".Z")
    ]

    n = len(files)
    total += n

    if n != 72:
        print(
            f"DOY {doy:03d}: {n:2d}/72 "
            f"缺 {72-n:2d}"
        )

print(f"\n总文件数：{total}")
print(f"理论文件数：{91 * 72}")
print(f"总缺失数：{91 * 72 - total}")