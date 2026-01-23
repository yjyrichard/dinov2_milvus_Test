"""
外观专利 XML 解析器

解析 USPTO 外观专利 XML 文件，提取元数据和图片列表。
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class DesignPatent:
    """外观专利数据结构"""
    # 核心字段
    patent_id: str = ""                    # 专利号，如 D1107392
    kind: str = "S1"                       # 文献类型

    # 元数据
    title: str = ""                        # 设计名称
    loc_class: str = ""                    # LOC分类号
    loc_edition: str = ""                  # LOC版本
    pub_date: int = 0                      # 公开日期 YYYYMMDD
    filing_date: int = 0                   # 申请日期 YYYYMMDD
    grant_term: int = 15                   # 授权期限（年）

    # 当事人
    applicant_name: str = ""               # 申请人名称
    applicant_country: str = ""            # 申请人国家
    inventor_names: str = ""               # 发明人（逗号分隔）
    assignee_name: str = ""                # 受让人名称

    # 权利要求
    claim_text: str = ""                   # 权利要求文本

    # 图片
    images: list = field(default_factory=list)  # 图片文件名列表
    image_count: int = 0                   # 图片数量

    # 文件路径
    xml_path: str = ""                     # XML文件路径
    data_dir: str = ""                     # 数据目录

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


def safe_get_text(element, path: str, default: str = '') -> str:
    """安全获取XML元素文本"""
    if element is None:
        return default
    node = element.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return default


def safe_get_attr(element, path: str, attr: str, default: str = '') -> str:
    """安全获取XML元素属性"""
    if element is None:
        return default
    node = element.find(path)
    if node is not None:
        return node.get(attr, default)
    return default


def parse_date(date_str: str) -> int:
    """解析日期字符串 YYYYMMDD -> int"""
    if not date_str or len(date_str) != 8:
        return 0
    try:
        return int(date_str)
    except ValueError:
        return 0


def parse_applicant(biblio) -> tuple[str, str]:
    """
    解析申请人信息
    返回: (name, country)
    """
    applicant = biblio.find('.//us-applicants/us-applicant')
    if applicant is None:
        return '', ''

    addressbook = applicant.find('addressbook')
    if addressbook is None:
        return '', ''

    # 优先取公司名
    orgname = safe_get_text(addressbook, 'orgname')
    if orgname:
        country = safe_get_text(addressbook, 'address/country')
        return orgname, country

    # 其次取个人名
    first = safe_get_text(addressbook, 'first-name')
    last = safe_get_text(addressbook, 'last-name')
    name = f"{first} {last}".strip()
    country = safe_get_text(addressbook, 'address/country')

    return name, country


def parse_inventors(biblio) -> str:
    """
    解析发明人列表
    返回: 逗号分隔的发明人名称
    """
    inventors = biblio.findall('.//inventors/inventor')
    names = []

    for inv in inventors:
        addressbook = inv.find('addressbook')
        if addressbook is None:
            continue

        first = safe_get_text(addressbook, 'first-name')
        last = safe_get_text(addressbook, 'last-name')
        name = f"{first} {last}".strip()
        if name:
            names.append(name)

    return ', '.join(names)


def parse_assignee(biblio) -> str:
    """解析受让人名称"""
    assignee = biblio.find('.//assignees/assignee')
    if assignee is None:
        return ''

    addressbook = assignee.find('addressbook')
    if addressbook is None:
        return ''

    return safe_get_text(addressbook, 'orgname')


def parse_images(root) -> list[str]:
    """解析图片文件名列表"""
    images = []
    for img in root.findall('.//drawings/figure/img'):
        file_name = img.get('file')
        if file_name:
            images.append(file_name)
    return images


def parse_claim(root) -> str:
    """解析权利要求文本"""
    claim = root.find('.//claims/claim/claim-text')
    if claim is not None:
        # 获取所有文本（包括子元素）
        text = ''.join(claim.itertext())
        return text.strip()
    return ''


def parse_design_patent_xml(xml_path: str) -> Optional[DesignPatent]:
    """
    解析外观专利 XML 文件

    Args:
        xml_path: XML 文件路径

    Returns:
        DesignPatent 对象，解析失败返回 None
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 验证是外观专利
        if root.tag != 'us-patent-grant':
            print(f"[PARSER] 非外观专利XML: {xml_path}, 根元素: {root.tag}")
            return None

        # 书目数据
        biblio = root.find('us-bibliographic-data-grant')
        if biblio is None:
            print(f"[PARSER] 找不到书目数据: {xml_path}")
            return None

        # 公开信息
        pub_ref = biblio.find('publication-reference/document-id')
        app_ref = biblio.find('application-reference/document-id')

        # 解析各字段
        patent = DesignPatent()

        # 专利号（不含 US 前缀）
        patent.patent_id = safe_get_text(pub_ref, 'doc-number')
        patent.kind = safe_get_text(pub_ref, 'kind', 'S1')

        # 日期
        patent.pub_date = parse_date(safe_get_text(pub_ref, 'date'))
        patent.filing_date = parse_date(safe_get_text(app_ref, 'date'))

        # 标题
        patent.title = safe_get_text(biblio, 'invention-title')

        # LOC 分类
        loc = biblio.find('classification-locarno')
        if loc is not None:
            patent.loc_class = safe_get_text(loc, 'main-classification')
            patent.loc_edition = safe_get_text(loc, 'edition')

        # 授权期限
        grant_term = safe_get_text(biblio, 'us-term-of-grant/length-of-grant')
        if grant_term:
            try:
                patent.grant_term = int(grant_term)
            except ValueError:
                patent.grant_term = 15

        # 申请人
        patent.applicant_name, patent.applicant_country = parse_applicant(biblio)

        # 发明人
        patent.inventor_names = parse_inventors(biblio)

        # 受让人
        patent.assignee_name = parse_assignee(biblio)

        # 权利要求
        patent.claim_text = parse_claim(root)[:500]  # 限制长度

        # 图片
        patent.images = parse_images(root)
        patent.image_count = len(patent.images)

        # 文件路径
        patent.xml_path = xml_path
        patent.data_dir = str(Path(xml_path).parent)

        return patent

    except ET.ParseError as e:
        print(f"[PARSER] XML解析错误: {xml_path}, {e}")
        return None
    except Exception as e:
        print(f"[PARSER] 解析失败: {xml_path}, {e}")
        import traceback
        traceback.print_exc()
        return None


def scan_design_patents(data_dir: str) -> list[DesignPatent]:
    """
    扫描目录下的所有外观专利

    Args:
        data_dir: 数据目录（包含 USD* 子目录）

    Returns:
        DesignPatent 列表
    """
    data_path = Path(data_dir)
    patents = []

    # 查找所有 USD* 目录
    for patent_dir in data_path.glob('USD*'):
        if not patent_dir.is_dir():
            continue

        # 查找 XML 文件
        xml_files = list(patent_dir.glob('*.XML'))
        if not xml_files:
            continue

        xml_path = xml_files[0]
        patent = parse_design_patent_xml(str(xml_path))

        if patent:
            patents.append(patent)
            print(f"[SCAN] 解析成功: {patent.patent_id} - {patent.title[:30]}...")
        else:
            print(f"[SCAN] 解析失败: {xml_path}")

    print(f"[SCAN] 共扫描 {len(patents)} 个外观专利")
    return patents


def scan_design_patents_nested(design_dir: str) -> Iterator[DesignPatent]:
    """
    扫描嵌套目录结构的外观专利 (生成器模式，节省内存)

    目录结构: DESIGN/USD*-日期/USD*-日期/*.XML

    Args:
        design_dir: DESIGN 目录路径

    Yields:
        DesignPatent 对象
    """
    design_path = Path(design_dir)

    # 查找所有 USD* 目录
    for patent_dir in design_path.glob('USD*'):
        if not patent_dir.is_dir():
            continue

        # 嵌套结构: USD*/USD*/*.XML
        xml_files = list(patent_dir.glob('*/*.XML'))
        if not xml_files:
            # 也尝试直接在当前目录找
            xml_files = list(patent_dir.glob('*.XML'))

        if not xml_files:
            continue

        xml_path = xml_files[0]
        patent = parse_design_patent_xml(str(xml_path))

        if patent:
            yield patent
        else:
            print(f"[SCAN] 解析失败: {xml_path}")


def scan_all_design_patents(root_dir: str, verbose: bool = True) -> Iterator[DesignPatent]:
    """
    扫描根目录下所有日期目录中的 DESIGN 子目录 (生成器模式，节省内存)

    目录结构: D:\\data\\I日期\\I日期\\DESIGN\\USD*\\USD*\\*.XML

    Args:
        root_dir: 根目录路径 (如 D:\\data)
        verbose: 是否输出详细信息

    Yields:
        DesignPatent 对象，逐个返回避免内存压力
    """
    root_path = Path(root_dir)

    # 查找所有 DESIGN 目录: I*/I*/DESIGN
    design_dirs = list(root_path.glob('I*/I*/DESIGN'))

    if verbose:
        print(f"[SCAN] 找到 {len(design_dirs)} 个 DESIGN 目录")

    patent_count = 0
    for idx, design_dir in enumerate(design_dirs):
        if verbose:
            print(f"\n[{idx + 1}/{len(design_dirs)}] 扫描: {design_dir}")

        dir_count = 0
        for patent in scan_design_patents_nested(str(design_dir)):
            yield patent
            patent_count += 1
            dir_count += 1

        if verbose:
            print(f"  找到 {dir_count} 个专利")

    if verbose:
        print(f"\n[SCAN] 总计: {patent_count} 个外观专利")


# 测试代码
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--scan-all":
            # 扫描 D:\data 下所有 DESIGN 目录
            root_dir = sys.argv[2] if len(sys.argv) > 2 else r"D:\data"
            print(f"扫描目录: {root_dir}")
            patents = scan_all_design_patents(root_dir)

            print(f"\n=== 扫描结果 (前10个) ===")
            for patent in patents[:10]:
                print(f"{patent.patent_id}: {patent.title[:40]} (LOC: {patent.loc_class}, 图片: {patent.image_count})")

            # 统计图片总数
            total_images = sum(p.image_count for p in patents)
            print(f"\n总计: {len(patents)} 个专利, {total_images} 张图片")

        elif arg.endswith('.XML') or arg.endswith('.xml'):
            # 测试单个文件
            patent = parse_design_patent_xml(arg)
            if patent:
                print("\n=== 解析结果 ===")
                for key, value in patent.to_dict().items():
                    if key != 'images':
                        print(f"{key}: {value}")
                print(f"images: {patent.images[:3]}..." if len(patent.images) > 3 else f"images: {patent.images}")
        else:
            # 扫描指定目录
            patents = scan_design_patents(arg)
            print(f"\n=== 扫描结果 ===")
            for patent in patents[:5]:
                print(f"{patent.patent_id}: {patent.title[:40]} (LOC: {patent.loc_class}, 图片: {patent.image_count})")
    else:
        print("用法:")
        print("  python design_patent_parser.py <xml_file>       - 解析单个XML")
        print("  python design_patent_parser.py <data_dir>       - 扫描目录")
        print("  python design_patent_parser.py --scan-all [dir] - 扫描所有DESIGN目录 (默认 D:\\data)")
