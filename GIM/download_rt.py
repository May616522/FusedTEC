import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 配置
# ============================================================

YEAR = 2021

# 约 3 个月：2021-04-01 ~ 2021-06-30
START_DOY = 91
END_DOY = 181

OUTPUT_DIR = r"F:\FusedTec\Data\GIM2021\IGS\RT"

BASE_URL = "https://chapman.upc.es/irtg/archive"


# ============================================================
# requests session
# ============================================================

def create_session():

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        )
    })

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# ============================================================
# 获取某一天所有 IONEX 文件
# ============================================================

def get_daily_files(session, year, doy):

    doy_str = f"{doy:03d}"

    url = (
        f"{BASE_URL}/"
        f"{year}/"
        f"{doy_str}/"
        f"global_vtec_snapshots_since_last_midnight/"
        f"ionex_file/"
    )

    print()
    print("=" * 80)
    print(f"读取目录：YEAR={year}, DOY={doy_str}")
    print(url)

    try:

        response = session.get(
            url,
            timeout=(20, 60)
        )

        if response.status_code == 404:

            print("目录不存在，跳过")
            return url, []

        response.raise_for_status()

    except requests.RequestException as e:

        print(f"读取目录失败：{e}")
        return url, []

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    files = []

    for a in soup.find_all("a"):

        href = a.get("href")

        if not href:
            continue

        # 只下载 IRTG IONEX 压缩文件
        if (
            href.lower().startswith("irtg")
            and href.endswith(".Z")
        ):
            files.append(href)

    files = sorted(set(files))

    print(f"发现 {len(files)} 个 IONEX 文件")

    return url, files


# ============================================================
# 下载单个文件
# ============================================================

def download_file(
    session,
    file_url,
    output_path
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # --------------------------------------------------------
    # 已下载自动跳过
    # --------------------------------------------------------

    if (
        os.path.exists(output_path)
        and os.path.getsize(output_path) > 1000
    ):

        print(
            f"[SKIP] 已存在："
            f"{os.path.basename(output_path)}"
        )

        return True

    temp_path = output_path + ".part"

    try:

        with session.get(
            file_url,
            stream=True,
            timeout=(20, 120)
        ) as response:

            if response.status_code == 404:

                print(
                    f"[404] "
                    f"{os.path.basename(output_path)}"
                )

                return False

            response.raise_for_status()

            with open(temp_path, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 64
                ):

                    if chunk:
                        f.write(chunk)

        # 检查文件
        if os.path.getsize(temp_path) < 1000:

            print(
                f"[ERROR] 文件异常小："
                f"{os.path.basename(output_path)}"
            )

            os.remove(temp_path)

            return False

        os.replace(
            temp_path,
            output_path
        )

        print(
            f"[OK] "
            f"{os.path.basename(output_path)}"
        )

        return True

    except requests.RequestException as e:

        print(
            f"[ERROR] "
            f"{os.path.basename(output_path)}: "
            f"{e}"
        )

        if os.path.exists(temp_path):

            try:
                os.remove(temp_path)
            except OSError:
                pass

        return False


# ============================================================
# 主程序
# ============================================================

def main():

    session = create_session()

    total_success = 0
    total_skip_or_success = 0
    total_failed = 0

    for doy in range(
        START_DOY,
        END_DOY + 1
    ):

        doy_str = f"{doy:03d}"

        daily_url, files = get_daily_files(
            session,
            YEAR,
            doy
        )

        if not files:

            print(
                f"DOY {doy_str} "
                f"没有发现可下载数据"
            )

            continue

        # 每天单独建目录
        daily_dir = os.path.join(
            OUTPUT_DIR,
            str(YEAR),
            doy_str
        )

        day_ok = 0
        day_fail = 0

        for filename in files:

            file_url = urljoin(
                daily_url,
                filename
            )

            output_path = os.path.join(
                daily_dir,
                filename
            )

            success = download_file(
                session,
                file_url,
                output_path
            )

            if success:

                day_ok += 1
                total_skip_or_success += 1

            else:

                day_fail += 1
                total_failed += 1

            # 不要过快访问服务器
            time.sleep(0.2)

        print(
            f"\nDOY {doy_str} 完成："
            f"成功/已存在 {day_ok}, "
            f"失败 {day_fail}"
        )

    print()
    print("=" * 80)
    print("全部下载结束")
    print(
        f"成功/已存在："
        f"{total_skip_or_success}"
    )
    print(
        f"失败："
        f"{total_failed}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()