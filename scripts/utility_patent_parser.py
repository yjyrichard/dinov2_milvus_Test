"""
发明专利 XML 解析器
支持 A1（申请公开）、B2（授权专利）、E1（再颁专利）

目录结构:
D:\data\I日期\I日期\UTIL\专利号\专利号\*.XML
D:\data\I日期\I日期\REISSUE\专利号\专利号\*.XML
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, List, Generator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class UtilityPatent:
    """发明专利数据结构"""
    # 基础标识
    application_number: str = ""
    publication_number: str = ""
    patent_type: str = ""  # A1/B1/B2/E1

    # 核心文本
    title: str = ""
    abstract: str = ""
    claims: str = ""  # 合并的权利要求文本
    independent_claims: List[str] = field(default_factory=list)  # 独立权利要求列表

    # 日期
    publication_date: int = 0  # YYYYMMDD
    application_date: int = 0

    # 当事人
    assignee: str = ""
    assignee_country: str = ""
    inventors: List[str] = field(default_factory=list)

    # 分类号
    main_cpc: str = ""
    all_cpc_codes: List[str] = field(default_factory=list)
    main_ipc: str = ""
    all_ipc_codes: List[str] = field(default_factory=list)

    # B2/E1 特有
    term_extension: int = 0
    examiner: str = ""
    number_of_claims: int = 0
    related_publication: str = ""

    # E1 特有
    original_patent_number: str = ""
    original_application_number: str = ""

    # 元数据
    source_type: str = ""  # UTIL/REISSUE
    xml_file: str = ""
    data_dir: str = ""


def parse_date(date_str: str) -> int:
    """解析日期字符串为整数 YYYYMMDD"""
    if date_str and len(date_str) >= 8:
        try:
            return int(date_str[:8])
        except ValueError:
            pass
    return 0


def extract_text(element) -> str:
    """提取元素的所有文本内容"""
    if element is None:
        return ""
    return ''.join(element.itertext()).strip()


def parse_utility_patent_xml(xml_path: str) -> Optional[UtilityPatent]:
    """
    解析发明专利 XML 文件
    自动识别 A1/B2/E1 类型

    Args:
        xml_path: XML文件路径

    Returns:
        UtilityPatent 对象，解析失败返回 None
    """
    try:
        # 解析 XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        root_tag = root.tag

        # 确定专利类型和书目数据节点
        if root_tag == 'us-patent-application':
            patent_type = 'A1'
            biblio = root.find('us-bibliographic-data-application')
        elif root_tag == 'us-patent-grant':
            biblio = root.find('us-bibliographic-data-grant')

            # 根据 appl-type 区分 B2 和 E1
            app_ref = biblio.find('application-reference') if biblio is not None else None
            appl_type = app_ref.get('appl-type', '') if app_ref is not None else ''

            if appl_type == 'reissue':
                patent_type = 'E1'
            else:
                # 从 kind code 获取 B1 或 B2
                kind = biblio.findtext('.//publication-reference/document-id/kind', 'B2') if biblio is not None else 'B2'
                patent_type = kind
        else:
            logger.warning(f"未知的根节点: {root_tag}, 文件: {xml_path}")
            return None

        if biblio is None:
            logger.warning(f"未找到书目数据节点: {xml_path}")
            return None

        # ============ 解析公共字段 ============
        # 公开/授权号
        pub_ref = biblio.find('.//publication-reference/document-id')
        publication_number = pub_ref.findtext('doc-number', '') if pub_ref is not None else ''
        publication_date = parse_date(pub_ref.findtext('date', '') if pub_ref is not None else '')

        # 申请号
        app_ref = biblio.find('.//application-reference/document-id')
        application_number = app_ref.findtext('doc-number', '') if app_ref is not None else ''
        application_date = parse_date(app_ref.findtext('date', '') if app_ref is not None else '')

        # 发明名称
        title = biblio.findtext('invention-title', '') or ''
        # 清理标题中的换行符
        title = ' '.join(title.split())

        # 摘要
        abstract_elem = root.find('.//abstract')
        abstract = extract_text(abstract_elem)

        # 权利要求
        claims_list = []
        independent_claims = []
        for claim in root.findall('.//claim'):
            claim_text = extract_text(claim)
            if claim_text:
                claims_list.append(claim_text)
                # 独立权利要求通常不包含 "according to claim" 等引用语
                claim_lower = claim_text.lower()
                if ('according to claim' not in claim_lower and
                    'as claimed in claim' not in claim_lower and
                    'of claim' not in claim_lower and
                    'the method of claim' not in claim_lower and
                    'the system of claim' not in claim_lower and
                    'the apparatus of claim' not in claim_lower):
                    independent_claims.append(claim_text)

        claims = '\n\n'.join(claims_list)

        # 专利权人
        assignee = ''
        assignee_country = ''
        assignee_elem = biblio.find('.//assignee')
        if assignee_elem is not None:
            assignee = assignee_elem.findtext('.//orgname', '')
            if not assignee:
                # 个人申请人
                first = assignee_elem.findtext('.//first-name', '')
                last = assignee_elem.findtext('.//last-name', '')
                assignee = f"{first} {last}".strip()
            addr = assignee_elem.find('.//address')
            if addr is not None:
                assignee_country = addr.findtext('country', '')

        # 发明人
        inventors = []
        for inventor in biblio.findall('.//inventor'):
            first = inventor.findtext('.//first-name', '')
            last = inventor.findtext('.//last-name', '')
            name = f"{first} {last}".strip()
            if name:
                inventors.append(name)

        # CPC 分类号
        cpc_codes = []
        for cpc in biblio.findall('.//classification-cpc'):
            section = cpc.findtext('section', '')
            cls = cpc.findtext('class', '')
            subclass = cpc.findtext('subclass', '')
            main_group = cpc.findtext('main-group', '')
            subgroup = cpc.findtext('subgroup', '')
            cpc_code = f"{section}{cls}{subclass}{main_group}/{subgroup}"
            if cpc_code.strip('/'):
                cpc_codes.append(cpc_code)

        main_cpc = cpc_codes[0] if cpc_codes else ''

        # IPC 分类号
        ipc_codes = []
        for ipc in biblio.findall('.//classification-ipcr'):
            section = ipc.findtext('section', '')
            cls = ipc.findtext('class', '')
            subclass = ipc.findtext('subclass', '')
            main_group = ipc.findtext('main-group', '')
            subgroup = ipc.findtext('subgroup', '')
            ipc_code = f"{section}{cls}{subclass}{main_group}/{subgroup}"
            if ipc_code.strip('/'):
                ipc_codes.append(ipc_code)

        main_ipc = ipc_codes[0] if ipc_codes else ''

        # ============ 解析 B2/E1 特有字段 ============
        term_extension = 0
        examiner = ''
        number_of_claims = 0
        related_publication = ''
        original_patent_number = ''
        original_application_number = ''

        if patent_type in ('B1', 'B2', 'E1'):
            # 专利期限延长
            term = biblio.find('.//us-term-of-grant/us-term-extension')
            if term is not None and term.text:
                try:
                    term_extension = int(term.text)
                except ValueError:
                    pass

            # 审查员
            exam = biblio.find('.//primary-examiner')
            if exam is not None:
                first = exam.findtext('first-name', '')
                last = exam.findtext('last-name', '')
                examiner = f"{first} {last}".strip()

            # 权利要求数量
            num_claims = biblio.findtext('number-of-claims', '0')
            try:
                number_of_claims = int(num_claims)
            except ValueError:
                number_of_claims = len(claims_list)

            # 关联公开号 (B2 -> A1)
            related = biblio.find('.//us-related-documents/related-publication/document-id/doc-number')
            if related is not None and related.text:
                related_publication = related.text

        # ============ E1 特有字段 ============
        if patent_type == 'E1':
            reissue = biblio.find('.//us-related-documents/reissue')
            if reissue is not None:
                orig_app = reissue.find('.//parent-doc/document-id/doc-number')
                if orig_app is not None and orig_app.text:
                    original_application_number = orig_app.text

                orig_patent = reissue.find('.//parent-grant-document/document-id/doc-number')
                if orig_patent is not None and orig_patent.text:
                    original_patent_number = orig_patent.text

        return UtilityPatent(
            application_number=application_number,
            publication_number=publication_number,
            patent_type=patent_type,
            title=title,
            abstract=abstract,
            claims=claims,
            independent_claims=independent_claims,
            publication_date=publication_date,
            application_date=application_date,
            assignee=assignee,
            assignee_country=assignee_country,
            inventors=inventors,
            main_cpc=main_cpc,
            all_cpc_codes=cpc_codes,
            main_ipc=main_ipc,
            all_ipc_codes=ipc_codes,
            term_extension=term_extension,
            examiner=examiner,
            number_of_claims=number_of_claims,
            related_publication=related_publication,
            original_patent_number=original_patent_number,
            original_application_number=original_application_number,
            xml_file=xml_path
        )

    except ET.ParseError as e:
        logger.error(f"XML 解析错误 {xml_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"解析异常 {xml_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def scan_all_utility_patents(
    data_dir: str,
    source_types: List[str] = None
) -> Generator[UtilityPatent, None, None]:
    """
    扫描所有发明专利（生成器模式）

    目录结构:
    D:\data\I日期\I日期\UTIL\专利号\专利号\*.XML
    D:\data\I日期\I日期\REISSUE\专利号\专利号\*.XML

    排除规则:
    - 后缀为 -SUPP 的目录
    - DESIGN 目录

    Args:
        data_dir: 数据根目录，如 D:\data
        source_types: 要处理的类型列表，默认 ['UTIL', 'REISSUE']

    Yields:
        UtilityPatent 对象
    """
    if source_types is None:
        source_types = ['UTIL', 'REISSUE']

    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return

    # 遍历日期目录
    for date_dir in sorted(data_path.iterdir()):
        if not date_dir.is_dir():
            continue

        # 排除 -SUPP 目录
        if date_dir.name.endswith('-SUPP'):
            logger.debug(f"跳过 SUPP 目录: {date_dir.name}")
            continue

        # 内层同名目录
        inner_dir = date_dir / date_dir.name
        if not inner_dir.exists():
            # 尝试直接在当前目录下查找
            inner_dir = date_dir

        for source_type in source_types:
            type_dir = inner_dir / source_type
            if not type_dir.exists():
                continue

            logger.info(f"扫描目录: {type_dir}")

            # 遍历专利目录
            for patent_dir_1 in type_dir.iterdir():
                if not patent_dir_1.is_dir():
                    continue

                # 尝试二层目录结构: 专利号\专利号\*.XML
                patent_dir_2 = patent_dir_1 / patent_dir_1.name
                if patent_dir_2.exists() and patent_dir_2.is_dir():
                    xml_dir = patent_dir_2
                else:
                    # 单层目录结构: 专利号\*.XML
                    xml_dir = patent_dir_1

                # 查找 XML 文件
                xml_files = list(xml_dir.glob('*.XML')) + list(xml_dir.glob('*.xml'))

                for xml_file in xml_files:
                    patent = parse_utility_patent_xml(str(xml_file))
                    if patent:
                        patent.source_type = source_type
                        patent.data_dir = str(xml_dir)
                        yield patent


def count_patents(data_dir: str, source_types: List[str] = None) -> dict:
    """
    统计专利数量（不解析内容）

    Returns:
        {'UTIL': xxx, 'REISSUE': xxx, 'total': xxx}
    """
    if source_types is None:
        source_types = ['UTIL', 'REISSUE']

    counts = {t: 0 for t in source_types}
    data_path = Path(data_dir)

    if not data_path.exists():
        return counts

    for date_dir in data_path.iterdir():
        if not date_dir.is_dir() or date_dir.name.endswith('-SUPP'):
            continue

        inner_dir = date_dir / date_dir.name
        if not inner_dir.exists():
            inner_dir = date_dir

        for source_type in source_types:
            type_dir = inner_dir / source_type
            if not type_dir.exists():
                continue

            for patent_dir_1 in type_dir.iterdir():
                if not patent_dir_1.is_dir():
                    continue

                patent_dir_2 = patent_dir_1 / patent_dir_1.name
                xml_dir = patent_dir_2 if patent_dir_2.exists() else patent_dir_1

                xml_count = len(list(xml_dir.glob('*.XML'))) + len(list(xml_dir.glob('*.xml')))
                if xml_count > 0:
                    counts[source_type] += 1

    counts['total'] = sum(counts.values())
    return counts


if __name__ == "__main__":
    # 测试解析
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    data_dir = r"D:\data"

    if len(sys.argv) > 1:
        # 解析单个文件
        xml_path = sys.argv[1]
        print(f"解析文件: {xml_path}")
        patent = parse_utility_patent_xml(xml_path)
        if patent:
            print(f"\n{'='*60}")
            print(f"专利号: {patent.publication_number}")
            print(f"类型: {patent.patent_type}")
            print(f"申请号: {patent.application_number}")
            print(f"标题: {patent.title}")
            print(f"专利权人: {patent.assignee} ({patent.assignee_country})")
            print(f"发明人: {', '.join(patent.inventors[:3])}...")
            print(f"申请日: {patent.application_date}")
            print(f"公开日: {patent.publication_date}")
            print(f"主CPC: {patent.main_cpc}")
            print(f"摘要: {patent.abstract[:200]}...")
            print(f"独立权利要求数: {len(patent.independent_claims)}")
            if patent.patent_type == 'E1':
                print(f"原始专利号: {patent.original_patent_number}")
            print(f"{'='*60}")
        else:
            print("解析失败")
    else:
        # 统计并扫描
        print(f"数据目录: {data_dir}")
        print("\n统计专利数量...")
        counts = count_patents(data_dir)
        print(f"UTIL: {counts.get('UTIL', 0)}")
        print(f"REISSUE: {counts.get('REISSUE', 0)}")
        print(f"总计: {counts.get('total', 0)}")

        print("\n扫描前5个专利...")
        for i, patent in enumerate(scan_all_utility_patents(data_dir)):
            print(f"\n[{i+1}] {patent.publication_number} ({patent.patent_type})")
            print(f"    标题: {patent.title[:50]}...")
            print(f"    专利权人: {patent.assignee}")
            if i >= 4:
                break
