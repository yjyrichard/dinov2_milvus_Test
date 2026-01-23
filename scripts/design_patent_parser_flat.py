"""
外观专利 XML 解析器 - 支持扁平目录结构

目录结构: data_dir/I日期/DESIGN/USD*/USD*/ *.XML
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, field, asdict


@dataclass
class DesignPatent:
    """外观专利数据结构"""
    patent_id: str = ""
    kind: str = "S1"
    title: str = ""
    loc_class: str = ""
    loc_edition: str = ""
    pub_date: int = 0
    filing_date: int = 0
    grant_term: int = 15
    applicant_name: str = ""
    applicant_country: str = ""
    inventor_names: str = ""
    assignee_name: str = ""
    claim_text: str = ""
    images: list = field(default_factory=list)
    image_count: int = 0
    xml_path: str = ""
    data_dir: str = ""
    image_dir: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def safe_get_text(element, path: str, default: str = '') -> str:
    if element is None:
        return default
    node = element.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return default


def parse_date(date_str: str) -> int:
    if not date_str or len(date_str) != 8:
        return 0
    try:
        return int(date_str)
    except ValueError:
        return 0


def parse_applicant(biblio) -> tuple[str, str]:
    applicant = biblio.find('.//us-applicants/us-applicant')
    if applicant is None:
        return '', ''

    addressbook = applicant.find('addressbook')
    if addressbook is None:
        return '', ''

    orgname = safe_get_text(addressbook, 'orgname')
    if orgname:
        country = safe_get_text(addressbook, 'address/country')
        return orgname, country

    first = safe_get_text(addressbook, 'first-name')
    last = safe_get_text(addressbook, 'last-name')
    name = f"{first} {last}".strip()
    country = safe_get_text(addressbook, 'address/country')

    return name, country


def parse_inventors(biblio) -> str:
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
    assignee = biblio.find('.//assignees/assignee')
    if assignee is None:
        return ''
    addressbook = assignee.find('addressbook')
    if addressbook is None:
        return ''
    return safe_get_text(addressbook, 'orgname')


def parse_images(root) -> list[str]:
    images = []
    for img in root.findall('.//drawings/figure/img'):
        file_name = img.get('file')
        if file_name:
            images.append(file_name)
    return images


def parse_claim(root) -> str:
    claim = root.find('.//claims/claim/claim-text')
    if claim is not None:
        text = ''.join(claim.itertext())
        return text.strip()
    return ''


def parse_design_patent_xml(xml_path: str) -> Optional[DesignPatent]:
    """解析单个外观专利 XML 文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if root.tag != 'us-patent-grant':
            print(f"[PARSER] 非外观专利XML: {xml_path}, 根元素: {root.tag}")
            return None

        biblio = root.find('us-bibliographic-data-grant')
        if biblio is None:
            print(f"[PARSER] 找不到书目数据: {xml_path}")
            return None

        pub_ref = biblio.find('publication-reference/document-id')
        app_ref = biblio.find('application-reference/document-id')

        patent = DesignPatent()
        patent.patent_id = safe_get_text(pub_ref, 'doc-number')
        patent.kind = safe_get_text(pub_ref, 'kind', 'S1')
        patent.pub_date = parse_date(safe_get_text(pub_ref, 'date'))
        patent.filing_date = parse_date(safe_get_text(app_ref, 'date'))
        patent.title = safe_get_text(biblio, 'invention-title')

        loc = biblio.find('classification-locarno')
        if loc is not None:
            patent.loc_class = safe_get_text(loc, 'main-classification')
            patent.loc_edition = safe_get_text(loc, 'edition')

        grant_term = safe_get_text(biblio, 'us-term-of-grant/length-of-grant')
        if grant_term:
            try:
                patent.grant_term = int(grant_term)
            except ValueError:
                patent.grant_term = 15

        patent.applicant_name, patent.applicant_country = parse_applicant(biblio)
        patent.inventor_names = parse_inventors(biblio)
        patent.assignee_name = parse_assignee(biblio)
        patent.claim_text = parse_claim(root)[:500]

        patent.images = parse_images(root)
        patent.image_count = len(patent.images)

        patent.xml_path = xml_path
        patent.data_dir = str(Path(xml_path).parent)
        patent.image_dir = str(Path(xml_path).parent)  # 图片和 XML 同目录

        return patent

    except ET.ParseError as e:
        print(f"[PARSER] XML解析错误: {xml_path}, {e}")
        return None
    except Exception as e:
        print(f"[PARSER] 解析失败: {xml_path}, {e}")
        import traceback
        traceback.print_exc()
        return None


def scan_design_patents_nested(design_dir: str) -> Iterator[DesignPatent]:
    """
    扫描嵌套目录结构的外观专利

    目录结构: DESIGN/USD*/USD*/ *.XML
    图片路径: USD*/USD*/ 目录
    """
    design_path = Path(design_dir)

    for patent_dir in design_path.glob('USD*'):
        if not patent_dir.is_dir():
            continue

        # USD*/USD*/*.XML
        xml_files = list(patent_dir.glob('*/*.XML'))
        if not xml_files:
            xml_files = list(patent_dir.glob('*/*.xml'))
        if not xml_files:
            # 也尝试直接在当前目录找
            xml_files = list(patent_dir.glob('*.XML'))
        if not xml_files:
            xml_files = list(patent_dir.glob('*.xml'))

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
    扫描根目录下所有外观专利

    目录结构: root_dir/I日期/DESIGN/USD*/USD*/ *.XML
    """
    root_path = Path(root_dir)

    # 查找所有 DESIGN 目录: I*/DESIGN
    design_dirs = list(root_path.glob('I*/DESIGN'))

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
        print(f"扫描目录: {root_dir}\n")

        patents = scan_all_design_patents(root_dir)

        count = 0
        for patent in patents:
            count += 1
            print(f"{count}. {patent.patent_id}: {patent.title[:40] if patent.title else 'N/A'}")
            print(f"   XML: {patent.xml_path}")
            print(f"   图片目录: {patent.image_dir}")
            print(f"   图片: {patent.images[:3]}..." if len(patent.images) > 3 else f"   图片: {patent.images}")

            # 验证图片是否存在
            for img in patent.images[:2]:
                img_path = os.path.join(patent.image_dir, img)
                exists = os.path.exists(img_path)
                status = '存在' if exists else '不存在'
                print(f"   {img}: {status}")
            print()

            if count >= 5:
                print("... (前5个)")
                break
    else:
        print("用法: python design_patent_parser_flat.py <根目录>")
