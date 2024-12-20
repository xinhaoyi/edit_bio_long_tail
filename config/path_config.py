from pathlib import Path
from dataclasses import dataclass
import os

ROOT_FLAG = '.root'

def find_root(current_path: Path) -> Path:
    """ Recursively find the project root directory flag file 'ROOT_FLAG' """
    # Check if the current path is the root directory
    if (current_path / ROOT_FLAG).exists():
        return current_path
    # If the current path cannot be traversed further into the parent directory, then raise FileNotFound
    if current_path.parent == current_path:
        raise FileNotFoundError(f"Project root flag file not found: {ROOT_FLAG}")
    # Otherwise, look recursively in the parent directory
    return find_root(current_path.parent)

def append_to_filename(path, content_to_append):
    """
    在文件名的末尾（扩展名之前）添加指定的内容。

    参数:
    path (Path or str): 原始文件路径，可以是 Path 对象或字符串。
    content_to_append (str): 要添加到文件名末尾的内容。

    返回:
    Path or str: 修改后的文件路径，类型与输入的 path 参数相同。
    """
    # 检查路径是否为 Path 对象
    if isinstance(path, Path):
        base_name = path.stem
        ext = path.suffix
        new_path = path.with_name(f"{base_name}{content_to_append}{ext}")
    else:
        # 处理字符串路径
        path_obj = Path(path)
        base_name = path_obj.stem
        ext = path_obj.suffix
        new_path_str = f"{base_name}{content_to_append}{ext}"
        # 重建完整的路径字符串
        new_path = str(path_obj.with_name(new_path_str))
    
    # 根据输入类型返回相应类型的结果
    return new_path if isinstance(path, Path) else new_path


@dataclass
class PathConfig:
    BASE_DIR = find_root(Path(__file__).resolve().parent)

    DATA_DIR = BASE_DIR / 'data'
    
    """knowledge triples"""

    KNOWLEDGE_TRIPLES_DATA_DIR = DATA_DIR / 'knowledge_triples'
    
    KNOWLEDGE_TRIPLES_RAW_DATA_FILE_NAME = 'clinic_knowledge_triples.json'
    
    KNOWLEDGE_TRIPLES_UPDATED_DATA_FILE_NAME = 'umls_knowledge_triples_updated.json'
    
    """knowledge triples"""

    PUBTATOR_DATA_DIR = DATA_DIR / 'pubtator'

    PUBTATOR_2_SNOMEDCT_DATA_DIR = DATA_DIR / 'pubtator2snomedCT'
    
    PUBTATOR_2_UMLS_DATA_DIR = DATA_DIR / 'pubtator2umls'

    UMLS_DATA_DIR = DATA_DIR / 'umls'

    SNOMEDCT_UMLS_DATA_DIR = UMLS_DATA_DIR / 'SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z'
    
    SNOMEDCT_UMLS_SAVE_DATA_DIR = SNOMEDCT_UMLS_DATA_DIR / 'process'

    METATHESAURUS_UMLS_DATA_DIR = UMLS_DATA_DIR / 'metathesaurus'
    
    METATHESAURUS_UMLS_SAVE_DATA_DIR = METATHESAURUS_UMLS_DATA_DIR / 'process'










