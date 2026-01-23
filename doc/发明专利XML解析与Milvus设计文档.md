# 发明专利XML解析与Milvus设计文档

## 一、概述

本文档详细说明USPTO发明专利三种类型（A1、B2、E1）的XML结构差异、解析方法以及Milvus向量数据库的设计方案。

### 1.1 专利类型说明

| Kind Code | 类型名称 | 中文名称 | 说明 |
|-----------|----------|----------|------|
| A1 | Application Publication | 申请公开 | 专利申请公开文本，尚未授权 |
| B2 | Granted Patent | 授权专利 | 经审查后授权的专利（有A1公开） |
| B1 | Granted Patent | 授权专利 | 直接授权的专利（无A1公开） |
| E1 | Reissue Patent | 再颁专利 | 对已授权专利的修正重新颁发 |

### 1.2 专利生命周期

![image-20260115113529241](D:\work\20251230需求分析\发明专利XML解析与Milvus设计文档.assets\image-20260115113529241.png)

---

## 二、XML结构对比

### 2.1 根节点差异

这是区分专利类型的**第一判断依据**：

| 专利类型 | 根节点 | DTD文件 |
|----------|--------|---------|
| **A1** | `<us-patent-application>` | us-patent-application-v46-2022-02-17.dtd |
| **B2/E1** | `<us-patent-grant>` | us-patent-grant-v47-2022-02-17.dtd |

### 2.2 书目数据节点差异

| 专利类型 | 书目数据节点 |
|----------|--------------|
| **A1** | `<us-bibliographic-data-application>` |
| **B2/E1** | `<us-bibliographic-data-grant>` |

### 2.3 申请类型属性（appl-type）

位于 `<application-reference>` 节点，这是区分B2和E1的**关键属性**：

```xml
<!-- A1 示例 -->
<application-reference appl-type="utility">

<!-- B2 示例 -->
<application-reference appl-type="utility">

<!-- E1 示例 -->
<application-reference appl-type="reissue">
```

---

## 三、详细字段对比

### 3.1 字段存在性对比表

| 字段/节点 | A1 | B2 | E1 | 说明 |
|-----------|:--:|:--:|:--:|------|
| `publication-reference` | ✓ | ✓ | ✓ | 公开/授权号 |
| `application-reference` | ✓ | ✓ | ✓ | 申请号（关联键） |
| `priority-claims` | ✓ | ✓ | ✓ | 优先权声明 |
| `classifications-ipcr` | ✓ | ✓ | ✓ | IPC分类 |
| `classifications-cpc` | ✓ | ✓ | ✓ | CPC分类 |
| `invention-title` | ✓ | ✓ | ✓ | 发明名称 |
| `us-parties` | ✓ | ✓ | ✓ | 当事人信息 |
| `assignees` | ✓ | ✓ | ✓ | 专利权人 |
| `abstract` | ✓ | ✓ | ✓ | 摘要 |
| `description` | ✓ | ✓ | ✓ | 说明书 |
| `claims` | ✓ | ✓ | ✓ | 权利要求 |
| `us-term-of-grant` | ✗ | ✓ | ✓ | 专利期限 |
| `us-references-cited` | ✗ | ✓ | ✓ | 引用文献 |
| `examiners` | ✗ | ✓ | ✓ | 审查员信息 |
| `agents` | ✗ | ✓ | ✓ | 代理人信息 |
| `number-of-claims` | ✗ | ✓ | ✓ | 权利要求数量 |
| `us-exemplary-claim` | ✗ | ✓ | ✓ | 示例性权利要求 |
| `us-related-documents` | 可选 | 可选 | **必有** | 关联文档 |
| `reissue` (子节点) | ✗ | ✗ | ✓ | 再颁信息 |

### 3.2 各类型XML结构详解

#### 3.2.1 A1（申请公开）结构

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE us-patent-application SYSTEM "us-patent-application-v46-2022-02-17.dtd">
<us-patent-application lang="EN" dtd-version="v4.6 2022-02-17"
    file="US20250361004A1-20251127.XML" status="PRODUCTION"
    id="us-patent-application" country="US"
    date-produced="20251111" date-publ="20251127">

    <us-bibliographic-data-application lang="EN" country="US">
        <!-- 公开号 -->
        <publication-reference>
            <document-id>
                <country>US</country>
                <doc-number>20250361004</doc-number>
                <kind>A1</kind>
                <date>20251127</date>
            </document-id>
        </publication-reference>

        <!-- 申请号（关联键） -->
        <application-reference appl-type="utility">
            <document-id>
                <country>US</country>
                <doc-number>18800713</doc-number>
                <date>20240812</date>
            </document-id>
        </application-reference>

        <us-application-series-code>18</us-application-series-code>
        <priority-claims>...</priority-claims>
        <classifications-ipcr>...</classifications-ipcr>
        <classifications-cpc>...</classifications-cpc>
        <invention-title>...</invention-title>
        <us-parties>...</us-parties>
        <assignees>...</assignees>
    </us-bibliographic-data-application>

    <abstract>...</abstract>
    <description>...</description>
    <claims>...</claims>
</us-patent-application>
```

**A1特点：**
- 根节点：`us-patent-application`
- `appl-type="utility"`
- **无** `us-term-of-grant`、`us-references-cited`、`examiners`、`agents`
- 权利要求通常较宽泛（未经审查修改）

#### 3.2.2 B2（授权专利）结构

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE us-patent-grant SYSTEM "us-patent-grant-v47-2022-02-17.dtd">
<us-patent-grant lang="EN" dtd-version="v4.7 2022-02-17"
    file="US12367005-20250722.XML" status="PRODUCTION"
    id="us-patent-grant" country="US"
    date-produced="20250708" date-publ="20250722">

    <us-bibliographic-data-grant>
        <!-- 授权号 -->
        <publication-reference>
            <document-id>
                <country>US</country>
                <doc-number>12367005</doc-number>
                <kind>B2</kind>
                <date>20250722</date>
            </document-id>
        </publication-reference>

        <!-- 申请号（关联键） -->
        <application-reference appl-type="utility">
            <document-id>
                <country>US</country>
                <doc-number>18460257</doc-number>
                <date>20230901</date>
            </document-id>
        </application-reference>

        <!-- B2特有：专利期限 -->
        <us-term-of-grant>
            <us-term-extension>103</us-term-extension>
        </us-term-of-grant>

        <!-- B2特有：引用文献 -->
        <us-references-cited>
            <us-citation>
                <patcit num="00001">...</patcit>
                <category>cited by examiner</category>
            </us-citation>
        </us-references-cited>

        <number-of-claims>19</number-of-claims>
        <us-exemplary-claim>1</us-exemplary-claim>

        <!-- 关联文档：指向A1公开 -->
        <us-related-documents>
            <related-publication>
                <document-id>
                    <country>US</country>
                    <doc-number>20230409273</doc-number>
                    <kind>A1</kind>
                </document-id>
            </related-publication>
        </us-related-documents>

        <!-- B2特有：审查员 -->
        <examiners>
            <primary-examiner>
                <last-name>Yang</last-name>
                <first-name>Kwang-Su</first-name>
                <department>2623</department>
            </primary-examiner>
        </examiners>
    </us-bibliographic-data-grant>

    <abstract>...</abstract>
    <description>...</description>
    <claims>...</claims>
</us-patent-grant>
```

**B2特点：**
- 根节点：`us-patent-grant`
- `appl-type="utility"`
- **有** `us-term-of-grant`（专利期限延长天数）
- **有** `us-references-cited`（审查员和申请人引用的文献）
- **有** `examiners`（审查员信息）
- **有** `related-publication`（指向对应的A1公开号）
- 权利要求经过审查修改，范围通常比A1窄

#### 3.2.3 E1（再颁专利）结构

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE us-patent-grant SYSTEM "us-patent-grant-v47-2022-02-17.dtd">
<us-patent-grant lang="EN" dtd-version="v4.7 2022-02-17"
    file="USRE050588-20250916.XML" status="PRODUCTION"
    id="us-patent-grant" country="US"
    date-produced="20250829" date-publ="20250916">

    <us-bibliographic-data-grant>
        <!-- 再颁号（RE开头） -->
        <publication-reference>
            <document-id>
                <country>US</country>
                <doc-number>RE050588</doc-number>
                <kind>E1</kind>
                <date>20250916</date>
            </document-id>
        </publication-reference>

        <!-- 关键区别：appl-type="reissue" -->
        <application-reference appl-type="reissue">
            <document-id>
                <country>US</country>
                <doc-number>18206463</doc-number>
                <date>20230606</date>
            </document-id>
        </application-reference>

        <us-term-of-grant>
            <us-term-extension>0</us-term-extension>
        </us-term-of-grant>

        <!-- E1特有：reissue节点，指向原始专利 -->
        <us-related-documents>
            <reissue>
                <relation>
                    <parent-doc>
                        <document-id>
                            <country>US</country>
                            <doc-number>16815811</doc-number>
                            <date>20200311</date>
                        </document-id>
                        <parent-status>GRANTED</parent-status>
                        <parent-grant-document>
                            <document-id>
                                <country>US</country>
                                <doc-number>11515138</doc-number>
                                <date>20221129</date>
                            </document-id>
                        </parent-grant-document>
                    </parent-doc>
                    <child-doc>
                        <document-id>
                            <country>US</country>
                            <doc-number>18206463</doc-number>
                        </document-id>
                    </child-doc>
                </relation>
            </reissue>
        </us-related-documents>
    </us-bibliographic-data-grant>
</us-patent-grant>
```

**E1特点：**
- 根节点：`us-patent-grant`（与B2相同）
- `appl-type="reissue"`（**关键区分点**）
- 专利号以 `RE` 开头
- **必有** `reissue` 节点，包含原始专利信息
- `parent-grant-document` 指向被替代的原始B2专利号

---

## 四、专利类型判断逻辑

### 4.1 判断流程图

```
                    ┌─────────────────────┐
                    │   读取XML文件        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   获取根节点名称     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────────┐     │     ┌──────────▼──────────┐
    │us-patent-application│    │     │  us-patent-grant    │
    └─────────┬─────────┘     │     └──────────┬──────────┘
              │               │                │
              │               │     ┌──────────▼──────────┐
              │               │     │ 获取appl-type属性   │
              │               │     └──────────┬──────────┘
              │               │                │
              │               │     ┌──────────┴──────────┐
              │               │     │                     │
    ┌─────────▼─────────┐     │  ┌──▼───┐           ┌────▼────┐
    │      A1类型       │     │  │utility│           │reissue  │
    └───────────────────┘     │  └──┬───┘           └────┬────┘
                              │     │                    │
                              │  ┌──▼───┐           ┌────▼────┐
                              │  │B1/B2 │           │   E1    │
                              │  └──────┘           └─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   其他类型(S1等)   │
                    └───────────────────┘
```

### 4.2 判断代码实现

```python
import xml.etree.ElementTree as ET

def detect_patent_type(xml_file_path: str) -> dict:
    """
    检测专利类型并返回相关信息

    Returns:
        dict: {
            'patent_type': 'A1' | 'B1' | 'B2' | 'E1' | 'unknown',
            'root_element': str,
            'appl_type': str,
            'kind_code': str
        }
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    root_tag = root.tag

    result = {
        'patent_type': 'unknown',
        'root_element': root_tag,
        'appl_type': None,
        'kind_code': None
    }

    # 第一步：根据根节点判断大类
    if root_tag == 'us-patent-application':
        # A1类型
        biblio = root.find('us-bibliographic-data-application')
        result['patent_type'] = 'A1'

    elif root_tag == 'us-patent-grant':
        # B2或E1类型，需要进一步判断
        biblio = root.find('us-bibliographic-data-grant')

        # 获取appl-type属性
        app_ref = biblio.find('application-reference')
        if app_ref is not None:
            appl_type = app_ref.get('appl-type')
            result['appl_type'] = appl_type

            if appl_type == 'reissue':
                result['patent_type'] = 'E1'
            elif appl_type == 'utility':
                # 进一步区分B1和B2
                kind = biblio.find('.//publication-reference/document-id/kind')
                if kind is not None:
                    result['kind_code'] = kind.text
                    result['patent_type'] = kind.text  # B1 或 B2

    return result
```

---

## 五、统一XML解析器

### 5.1 解析器设计思路

由于A1、B2、E1的XML结构存在差异，我们采用**策略模式**设计统一解析器：

1. **基础解析器**：处理三种类型共有的字段
2. **类型特定解析器**：处理各类型特有的字段
3. **统一入口**：自动检测类型并调用对应解析器

### 5.2 基础字段解析（三种类型通用）

```python
def parse_common_fields(root, biblio_node) -> dict:
    """解析三种类型共有的字段"""
    data = {}

    # 1. 公开/授权号
    pub_ref = biblio_node.find('.//publication-reference/document-id')
    if pub_ref is not None:
        data['publication_number'] = pub_ref.findtext('doc-number')
        data['kind_code'] = pub_ref.findtext('kind')
        data['publication_date'] = pub_ref.findtext('date')

    # 2. 申请号（关联键）
    app_ref = biblio_node.find('.//application-reference/document-id')
    if app_ref is not None:
        data['application_number'] = app_ref.findtext('doc-number')
        data['application_date'] = app_ref.findtext('date')

    # 3. 发明名称
    data['title'] = biblio_node.findtext('invention-title')

    # 4. IPC分类
    data['ipc_codes'] = []
    for ipc in biblio_node.findall('.//classification-ipcr'):
        section = ipc.findtext('section', '')
        cls = ipc.findtext('class', '')
        subclass = ipc.findtext('subclass', '')
        main_group = ipc.findtext('main-group', '')
        subgroup = ipc.findtext('subgroup', '')
        ipc_code = f"{section}{cls}{subclass}{main_group}/{subgroup}"
        data['ipc_codes'].append(ipc_code)

    # 5. CPC分类
    data['cpc_codes'] = []
    for cpc in biblio_node.findall('.//classification-cpc'):
        section = cpc.findtext('section', '')
        cls = cpc.findtext('class', '')
        subclass = cpc.findtext('subclass', '')
        main_group = cpc.findtext('main-group', '')
        subgroup = cpc.findtext('subgroup', '')
        cpc_code = f"{section}{cls}{subclass}{main_group}/{subgroup}"
        data['cpc_codes'].append(cpc_code)

    # 6. 专利权人
    data['assignees'] = []
    for assignee in biblio_node.findall('.//assignee'):
        org = assignee.findtext('.//orgname')
        if org:
            data['assignees'].append(org)

    # 7. 发明人
    data['inventors'] = []
    for inventor in biblio_node.findall('.//inventor'):
        first = inventor.findtext('.//first-name', '')
        last = inventor.findtext('.//last-name', '')
        data['inventors'].append(f"{first} {last}".strip())

    # 8. 摘要
    abstract = root.find('.//abstract')
    if abstract is not None:
        data['abstract'] = ''.join(abstract.itertext()).strip()

    # 9. 权利要求
    data['claims'] = []
    for claim in root.findall('.//claim'):
        claim_text = ''.join(claim.itertext()).strip()
        data['claims'].append(claim_text)

    return data
```

### 5.3 B2/E1特有字段解析

```python
def parse_grant_specific_fields(biblio_node) -> dict:
    """解析B2/E1特有的字段"""
    data = {}

    # 1. 专利期限延长
    term = biblio_node.find('.//us-term-of-grant/us-term-extension')
    data['term_extension'] = term.text if term is not None else '0'

    # 2. 权利要求数量
    num_claims = biblio_node.findtext('number-of-claims')
    data['number_of_claims'] = int(num_claims) if num_claims else 0

    # 3. 审查员信息
    examiner = biblio_node.find('.//primary-examiner')
    if examiner is not None:
        first = examiner.findtext('first-name', '')
        last = examiner.findtext('last-name', '')
        data['examiner'] = f"{first} {last}".strip()

    # 4. 引用文献
    data['citations'] = []
    for citation in biblio_node.findall('.//us-citation'):
        patcit = citation.find('patcit/document-id')
        if patcit is not None:
            doc_num = patcit.findtext('doc-number')
            if doc_num:
                data['citations'].append(doc_num)

    return data
```

### 5.4 E1特有字段解析

```python
def parse_reissue_specific_fields(biblio_node) -> dict:
    """解析E1再颁专利特有的字段"""
    data = {}

    # 查找reissue节点
    reissue = biblio_node.find('.//us-related-documents/reissue')
    if reissue is not None:
        # 原始申请号
        parent_app = reissue.find('.//parent-doc/document-id/doc-number')
        data['original_application_number'] = parent_app.text if parent_app is not None else None

        # 原始授权专利号
        parent_grant = reissue.find('.//parent-grant-document/document-id/doc-number')
        data['original_patent_number'] = parent_grant.text if parent_grant is not None else None

        # 原始授权日期
        parent_date = reissue.find('.//parent-grant-document/document-id/date')
        data['original_grant_date'] = parent_date.text if parent_date is not None else None

    return data
```

### 5.5 统一解析入口

```python
def parse_utility_patent(xml_file_path: str) -> dict:
    """
    统一解析入口，自动识别专利类型并解析
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    root_tag = root.tag

    # 初始化结果
    result = {'patent_type': 'unknown'}

    if root_tag == 'us-patent-application':
        # A1类型
        biblio = root.find('us-bibliographic-data-application')
        result['patent_type'] = 'A1'
        result.update(parse_common_fields(root, biblio))

    elif root_tag == 'us-patent-grant':
        biblio = root.find('us-bibliographic-data-grant')
        app_ref = biblio.find('application-reference')
        appl_type = app_ref.get('appl-type') if app_ref is not None else ''

        # 解析通用字段
        result.update(parse_common_fields(root, biblio))
        # 解析授权专利特有字段
        result.update(parse_grant_specific_fields(biblio))

        if appl_type == 'reissue':
            result['patent_type'] = 'E1'
            result.update(parse_reissue_specific_fields(biblio))
        else:
            kind = biblio.findtext('.//publication-reference/document-id/kind')
            result['patent_type'] = kind  # B1 或 B2

    return result
```

---

## 六、Milvus Collection设计

### 6.1 设计原则

1. **单一Collection存储所有发明专利类型**：A1、B2、E1存储在同一个Collection中
2. **使用申请号作为关联键**：便于去重和更新
3. **向量化核心文本字段**：摘要和权利要求
4. **标量字段支持过滤**：专利类型、分类号、日期等

### 6.2 Collection Schema设计（字段全集）

**设计原则**：包含A1/B2/E1所有字段的并集，某类型没有的字段存null/空。

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

utility_patent_schema = CollectionSchema(
    fields=[
        # ==================== 基础标识字段 ====================
        FieldSchema(name="application_number", dtype=DataType.VARCHAR, max_length=20,
                    is_primary=True, description="申请号（主键）"),
        FieldSchema(name="publication_number", dtype=DataType.VARCHAR, max_length=20,
                    description="公开号(A1)或授权号(B2/E1)"),
        FieldSchema(name="patent_type", dtype=DataType.VARCHAR, max_length=5,
                    description="专利类型：A1/B1/B2/E1"),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500,
                    description="发明名称"),

        # ==================== 日期字段 ====================
        FieldSchema(name="publication_date", dtype=DataType.VARCHAR, max_length=10,
                    description="公开/授权日期"),
        FieldSchema(name="application_date", dtype=DataType.VARCHAR, max_length=10,
                    description="申请日期"),

        # ==================== 申请人/专利权人信息 ====================
        FieldSchema(name="assignee", dtype=DataType.VARCHAR, max_length=500,
                    description="专利权人名称"),
        FieldSchema(name="assignee_country", dtype=DataType.VARCHAR, max_length=10,
                    description="专利权人国家（如CN/US/DE）"),
        FieldSchema(name="assignee_city", dtype=DataType.VARCHAR, max_length=100,
                    description="专利权人城市"),

        # ==================== 发明人信息 ====================
        FieldSchema(name="inventors", dtype=DataType.VARCHAR, max_length=2000,
                    description="发明人列表（逗号分隔）"),

        # ==================== 分类号 ====================
        FieldSchema(name="main_cpc", dtype=DataType.VARCHAR, max_length=20,
                    description="主CPC分类号"),
        FieldSchema(name="all_cpc_codes", dtype=DataType.VARCHAR, max_length=500,
                    description="所有CPC分类号（逗号分隔）"),
        FieldSchema(name="main_ipc", dtype=DataType.VARCHAR, max_length=20,
                    description="主IPC分类号"),
        FieldSchema(name="all_ipc_codes", dtype=DataType.VARCHAR, max_length=500,
                    description="所有IPC分类号（逗号分隔）"),

        # ==================== 优先权信息 ====================
        FieldSchema(name="priority_country", dtype=DataType.VARCHAR, max_length=10,
                    description="优先权国家"),
        FieldSchema(name="priority_number", dtype=DataType.VARCHAR, max_length=30,
                    description="优先权号"),
        FieldSchema(name="priority_date", dtype=DataType.VARCHAR, max_length=10,
                    description="优先权日期"),

        # ==================== 说明书分段存储 ====================
        FieldSchema(name="description_chunks", dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR, max_length=8000, max_capacity=100,
                    description="说明书分段文本（每段约8000字符）"),

        # ==================== 向量字段 ====================
        FieldSchema(name="title_vector", dtype=DataType.FLOAT_VECTOR, dim=768,
                    description="发明名称向量"),
        FieldSchema(name="abstract_vector", dtype=DataType.FLOAT_VECTOR, dim=768,
                    description="摘要文本向量"),
        FieldSchema(name="claims_vector", dtype=DataType.FLOAT_VECTOR, dim=768,
                    description="独立权利要求向量"),
        FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=768,
                    description="说明书向量（32k模型处理）"),

        # ==================== B2/E1特有字段（A1存空）====================
        FieldSchema(name="term_extension", dtype=DataType.VARCHAR, max_length=10,
                    description="专利期限延长天数"),
        FieldSchema(name="examiner", dtype=DataType.VARCHAR, max_length=100,
                    description="审查员姓名"),
        FieldSchema(name="examiner_department", dtype=DataType.VARCHAR, max_length=20,
                    description="审查员部门"),
        FieldSchema(name="number_of_claims", dtype=DataType.INT64,
                    description="权利要求数量"),
        FieldSchema(name="number_of_figures", dtype=DataType.INT64,
                    description="附图数量"),
        FieldSchema(name="agent", dtype=DataType.VARCHAR, max_length=200,
                    description="代理人/代理机构"),
        FieldSchema(name="related_publication", dtype=DataType.VARCHAR, max_length=20,
                    description="关联公开号（B2指向A1）"),
        FieldSchema(name="us_application_series_code", dtype=DataType.VARCHAR, max_length=5,
                    description="申请系列代码"),

        # ==================== E1特有字段（A1/B2存空）====================
        FieldSchema(name="original_patent_number", dtype=DataType.VARCHAR, max_length=20,
                    description="原始专利号（仅E1）"),
        FieldSchema(name="original_application_number", dtype=DataType.VARCHAR, max_length=20,
                    description="原始申请号（仅E1）"),
    ],
    description="发明专利Collection（A1/B2/E1统一存储，字段全集）"
)
```

**字段存储规则**：

| 字段 | A1 | B2 | E1 | 说明 |
|------|:--:|:--:|:--:|------|
| assignee_country | ✓ | ✓ | ✓ | 专利权人国家 |
| assignee_city | ✓ | ✓ | ✓ | 专利权人城市 |
| priority_country | ✓ | ✓ | ✓ | 优先权国家 |
| priority_number | ✓ | ✓ | ✓ | 优先权号 |
| priority_date | ✓ | ✓ | ✓ | 优先权日期 |
| term_extension | 空 | ✓ | ✓ | 专利期限延长 |
| examiner | 空 | ✓ | ✓ | 审查员 |
| examiner_department | 空 | ✓ | ✓ | 审查员部门 |
| number_of_claims | 0 | ✓ | ✓ | 权利要求数 |
| number_of_figures | 0 | ✓ | ✓ | 附图数 |
| agent | 空 | ✓ | ✓ | 代理人 |
| related_publication | 空 | ✓ | 空 | B2关联的A1号 |
| original_patent_number | 空 | 空 | ✓ | E1原始专利号 |
| original_application_number | 空 | 空 | ✓ | E1原始申请号 |

### 6.3 设计理由说明

#### 6.3.1 为什么使用申请号作为主键？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **申请号（推荐）** | A1和B2共享同一申请号，便于去重更新 | 需要额外存储公开/授权号 |
| 公开/授权号 | 直观，用户常用 | A1和B2号码不同，无法关联 |
| 自增ID | 简单 | 无业务意义，无法去重 |

**结论**：申请号是连接A1和B2的唯一纽带，使用它作为主键可以实现：
- 当B2入库时，自动覆盖对应的A1
- 当E1入库时，可以找到并删除原始B2

#### 6.3.2 为什么使用两个向量字段？

| 向量字段 | 用途 | 说明 |
|----------|------|------|
| `abstract_vector` | 技术领域相似度搜索 | 摘要描述发明的技术领域和解决的问题 |
| `claims_vector` | 侵权风险评估 | 权利要求定义专利的法律保护范围 |

**好处**：
1. **精准匹配**：用户可以选择搜索"技术相似"或"权利要求相似"
2. **侵权检测**：权利要求向量更适合判断产品是否落入专利保护范围
3. **灵活查询**：支持多向量联合搜索

#### 6.3.3 为什么不向量化说明书（Description）？

**说明书的特点**：
- 长度：通常 **5万-50万字符**，有的甚至更长
- 内容：详细技术描述，包含实施例、附图说明等

**不直接向量化的原因**：

| 问题 | 说明 |
|------|------|
| **长度超限** | Milvus VARCHAR最大65535字符，说明书远超此限制 |
| **模型限制** | BERT等模型token限制512，无法直接处理长文本 |
| **信息冗余** | 说明书与摘要内容重叠度高达80%+ |
| **存储成本** | 向量化后存储成本极高 |

**推荐方案：MySQL + Milvus 混合存储**

```
┌─────────────────────────────────────────────────────────┐
│                      MySQL                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │ patent_full_text 表                              │    │
│  │ - application_number (关联键)                    │    │
│  │ - abstract (TEXT)                                │    │
│  │ - description (LONGTEXT) ← 说明书全文            │    │
│  │ - claims (LONGTEXT) ← 权利要求全文               │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          │ application_number 关联
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      Milvus                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │ utility_patent Collection                        │    │
│  │ - application_number (主键)                      │    │
│  │ - abstract_vector (向量)                         │    │
│  │ - claims_vector (向量)                           │    │
│  │ - 其他标量字段...                                │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**查询流程**：
1. 用户输入产品描述 → 向量化
2. Milvus向量搜索 → 返回Top-K相似专利的application_number
3. 用MySQL查询完整说明书 → 展示给用户

#### 6.3.4 说明书向量化方案（32k Token模型）

**模型支持32k token，说明书处理策略**：

| 说明书长度 | 约Token数 | 处理方式 |
|------------|-----------|----------|
| < 12万字符 | < 32k | 直接整体向量化 |
| > 12万字符 | > 32k | 截取前32k token向量化 |

**说明书分段存储（description_chunks）**：

```python
def split_description(description: str, chunk_size: int = 8000) -> list:
    """
    将说明书分段存储
    - chunk_size: 每段约8000字符（约2k token）
    - 按段落分割，保持语义完整性
    """
    paragraphs = description.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks  # 存入description_chunks字段
```

**说明书向量化（description_vector）**：

```python
def vectorize_description(description: str, model) -> list:
    """
    使用32k token模型向量化说明书
    - 如果超过32k token，截取前面部分
    - 返回单个向量
    """
    # 32k token ≈ 12万字符（英文）
    max_chars = 120000
    text = description[:max_chars] if len(description) > max_chars else description

    vector = model.encode(text)
    return vector  # 存入description_vector字段
```

---

## 七、去重策略（Upsert机制）

### 7.1 为什么必须去重？

如果不去重，会产生以下问题：

| 问题 | 影响 |
|------|------|
| **数据冗余** | 同一技术的A1和B2都存在，搜索结果重复 |
| **法律误判** | A1的权利要求未经审查，范围宽泛，不代表实际保护范围 |
| **存储浪费** | A1和B2的说明书95%相同，向量几乎一致 |
| **搜索体验差** | Top-10结果可能有5个是同一专利的不同版本 |

### 7.2 去重优先级规则

```
优先级：E1 > B2 > B1 > A1
```

| 场景 | 操作 |
|------|------|
| 入库A1，无已有数据 | 直接插入 |
| 入库B2，已有A1 | 删除A1，插入B2 |
| 入库B2，无A1 | 直接插入（B1情况） |
| 入库E1，已有B2 | 删除B2，插入E1 |

### 7.3 去重实现代码

```python
from pymilvus import Collection

# 优先级映射
PRIORITY = {'A1': 1, 'B1': 2, 'B2': 3, 'E1': 4}

def upsert_patent(collection: Collection, patent_data: dict):
    """
    智能入库：根据优先级决定是否覆盖
    """
    app_num = patent_data['application_number']
    new_type = patent_data['patent_type']

    # 查询是否已存在
    results = collection.query(
        expr=f'application_number == "{app_num}"',
        output_fields=['patent_type']
    )

    if results:
        existing_type = results[0]['patent_type']

        # 比较优先级
        if PRIORITY.get(new_type, 0) <= PRIORITY.get(existing_type, 0):
            print(f"跳过：{app_num} 已有更高优先级版本 {existing_type}")
            return False

        # 删除旧版本
        collection.delete(f'application_number == "{app_num}"')
        print(f"删除旧版本：{app_num} ({existing_type})")

    # 插入新数据
    collection.insert([patent_data])
    print(f"插入：{app_num} ({new_type})")
    return True
```

### 7.4 E1的特殊处理

E1再颁专利需要额外处理：删除原始B2专利。

```python
def handle_reissue_patent(collection: Collection, e1_data: dict):
    """
    处理E1再颁专利：删除原始B2
    """
    original_patent = e1_data.get('original_patent_number')

    if original_patent:
        # 查找并删除原始B2
        results = collection.query(
            expr=f'publication_number == "{original_patent}"',
            output_fields=['application_number']
        )
        if results:
            old_app_num = results[0]['application_number']
            collection.delete(f'application_number == "{old_app_num}"')
            print(f"删除被替代的B2：{original_patent}")

    # 插入E1
    collection.insert([e1_data])
    print(f"插入E1：{e1_data['publication_number']}")
```

---

## 八、总结

### 8.1 三种专利类型快速对比

| 特征 | A1 | B2 | E1 |
|------|:--:|:--:|:--:|
| 根节点 | us-patent-application | us-patent-grant | us-patent-grant |
| appl-type | utility | utility | reissue |
| 法律效力 | 无（仅公开） | 有效 | 替代原B2 |
| 权利要求 | 宽泛 | 审查后确定 | 修正后确定 |
| 入库优先级 | 1（最低） | 3 | 4（最高） |

### 8.2 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 主键 | 申请号 | 唯一关联A1和B2，支持去重 |
| 向量字段 | 摘要+权利要求 | 分别支持技术搜索和侵权检测 |
| 去重策略 | E1>B2>A1 | 保留法律上最有效的版本 |
| Collection | 统一存储 | 简化查询，便于管理 |

### 8.3 去重带来的好处

1. **存储节省**：减少约40-50%的向量存储量
2. **搜索质量**：Top-K结果不再重复
3. **法律准确**：只展示有效的权利要求
4. **维护简单**：数据库始终保持最新状态

---

## 附录A：XML路径速查表

### 通用字段路径

| 字段 | A1路径 | B2/E1路径 |
|------|--------|-----------|
| 公开/授权号 | `us-bibliographic-data-application/publication-reference/document-id/doc-number` | `us-bibliographic-data-grant/publication-reference/document-id/doc-number` |
| 申请号 | `us-bibliographic-data-application/application-reference/document-id/doc-number` | `us-bibliographic-data-grant/application-reference/document-id/doc-number` |
| Kind Code | `us-bibliographic-data-application/publication-reference/document-id/kind` | `us-bibliographic-data-grant/publication-reference/document-id/kind` |
| 发明名称 | `us-bibliographic-data-application/invention-title` | `us-bibliographic-data-grant/invention-title` |
| 摘要 | `abstract` | `abstract` |
| 权利要求 | `claims/claim` | `claims/claim` |

### B2/E1特有字段路径

| 字段 | 路径 |
|------|------|
| 专利期限延长 | `us-bibliographic-data-grant/us-term-of-grant/us-term-extension` |
| 审查员 | `us-bibliographic-data-grant/examiners/primary-examiner` |
| 引用文献 | `us-bibliographic-data-grant/us-references-cited/us-citation` |
| 权利要求数量 | `us-bibliographic-data-grant/number-of-claims` |

### E1特有字段路径

| 字段 | 路径 |
|------|------|
| 原始申请号 | `us-bibliographic-data-grant/us-related-documents/reissue/relation/parent-doc/document-id/doc-number` |
| 原始专利号 | `us-bibliographic-data-grant/us-related-documents/reissue/relation/parent-doc/parent-grant-document/document-id/doc-number` |
| 原始授权日期 | `us-bibliographic-data-grant/us-related-documents/reissue/relation/parent-doc/parent-grant-document/document-id/date` |

---

## 附录B：示例数据

### A1示例（US20250361004A1）
- 公开号：20250361004
- 申请号：18800713
- 发明名称：METHOD AND SYSTEM OF AUTOMATIC WARNINGS...
- 专利权人：HONEYWELL INTERNATIONAL INC.

### B2示例（US12367005B2）
- 授权号：12367005
- 申请号：18460257
- 发明名称：Method for displaying application image...
- 专利权人：GUANGDONG OPPO MOBILE TELECOMMUNICATIONS CORP., LTD.
- 专利期限延长：103天

### E1示例（USRE050588E1）
- 再颁号：RE050588
- 申请号：18206463
- 发明名称：Ion trapping scheme with improved mass range
- 原始专利号：11515138
- 专利权人：Thermo Fisher Scientific (Bremen) GmbH
