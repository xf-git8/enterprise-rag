import pytesseract
from PIL import Image
from typing import List
import os, chardet, tempfile
import platform
import pdfplumber
from src.config import config
from src.utils import utils
from pdf2image import convert_from_path
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 60, enable_ocr: bool = True) -> None:
        """初始化文档处理器，配置核心参数和切割规则"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.enable_ocr = enable_ocr
        # 用于保证跨文件、跨批次切分时 chunk_id 的全局唯一性
        self._global_chunk_counter = 0

        # 调用本类内部方法获取路径
        self.poppler_path = self._get_poppler_path()
        self.tesseract_path = self._setup_tesseract_path()

    def _setup_tesseract_path(self):
        """智能配置 Tesseract OCR 引擎路径"""
        if platform.system() != "Windows":
            return  # Linux/Mac 依赖系统 PATH

        # 1. 优先从配置读取
        env_path = getattr(config, 'TESSERACT_BIN_PATH', None)
        if env_path and os.path.exists(env_path):
            pytesseract.pytesseract.tesseract_cmd = env_path
            print(f"从配置加载 Tesseract 路径: {env_path}")
            return

        # 2. 尝试常见的默认路径 (请根据你的实际安装情况调整)
        default_paths = [
            r"E:\damoxing\python-project\help_ocr\tesseract\install\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # 常见安装路径
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in default_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"自动发现 Tesseract 路径: {path}")
                return
        print("警告: 未在 Windows 上找到 Tesseract，OCR 功能将无法使用！")

    def _get_poppler_path(self):
        """智能获取 Poppler 路径"""
        if platform.system() != "Windows":
            return None

        # 1. 尝试从配置读取
        env_path = getattr(config, 'POPPLER_BIN_PATH', None)
        if env_path and os.path.exists(env_path):
            return env_path

        # 2. 尝试常见的默认路径
        default_paths = [
            r"E:\damoxing\python-project\help_ocr\poppler-26.02.0\Library\bin",
            r"C:\Program Files\poppler\Library\bin",  # 常见安装路径
            r".\poppler\Library\bin"
        ]
        for path in default_paths:
            # 检查目录下是否存在 pdftoppm.exe (Windows) 或 pdftoppm (Linux/Mac)
            exe_name = "pdftoppm.exe" if platform.system() == "Windows" else "pdftoppm"
            if os.path.exists(os.path.join(path, exe_name)):
                print(f"自动发现 Poppler 路径: {path}")
                return path
        print("警告: 未在 Windows 上找到 Poppler 路径，PDF 扫描件处理可能会失败。")
        return None

    def _extract_text_from_pdf_with_ocr(self, file_path: str, encoding: str) -> str:
        """智能提取 PDF 文本"""
        full_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                # 判断是否为扫描件
                sample_text = ""
                for page in pdf.pages[:3]:
                    text = page.extract_text()
                    if text:
                        sample_text += text

                # 【修复点】使用防御性清洗，防止 utils 报错
                try:
                    sample_text = utils._process_text(sample_text) if hasattr(utils,
                                                                              '_process_text') else sample_text.strip()
                except Exception as e:
                    sample_text = sample_text.strip()

                is_scanned = len(sample_text.strip()) < 50

                if not is_scanned:
                    # 原生文本提取
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            try:
                                page_text = utils._process_text(page_text) if hasattr(utils,
                                                                                      '_process_text') else page_text.strip()
                            except:
                                page_text = page_text.strip()
                            full_text += page_text + "\n\n"

                        # 图片 OCR
                        if self.enable_ocr:
                            try:
                                for img in page.images:
                                    # ... (crop logic) ...
                                    img_obj = page.crop((img["x0"], img["top"], img["x1"], img["bottom"])).to_image()
                                    ocr_text = pytesseract.image_to_string(img_obj.original, lang="chi_sim+eng")
                                    if ocr_text.strip():
                                        try:
                                            ocr_text = utils._process_text(ocr_text) if hasattr(utils,
                                                                                                '_process_text') else ocr_text.strip()
                                        except:
                                            ocr_text = ocr_text.strip()
                                        full_text += f"[图片OCR识别内容]:\n{ocr_text}\n\n"
                            except Exception as e:
                                print(f"页面图片OCR异常: {e}")
                else:
                    # 扫描件处理
                    if not self.enable_ocr:
                        raise ValueError("扫描件且未开启OCR")
                    print(f"检测到扫描件: {file_path}")
                    pages = convert_from_path(file_path, dpi=200, poppler_path=self.poppler_path)
                    for page_num, page_img in enumerate(pages, 1):
                        try:
                            ocr_text = pytesseract.image_to_string(page_img, lang="chi_sim+eng")
                            if ocr_text.strip():
                                full_text += f"=== 第{page_num}页 ===\n{ocr_text}\n\n"
                        except Exception as e:
                            print(f"扫描件第{page_num}页识别失败: {e}")
        except Exception as e:
            print(f"处理PDF失败 {file_path}: {e}")
        return full_text

    def _extract_text_from_doc_with_ocr(self, file_path: str, encoding: str) -> str:
        """提取 Word 文本"""
        full_text = ""
        try:
            doc = DocxDocument(file_path)
            # 段落
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text += paragraph.text + "\n\n"

            # 表格转 Markdown
            for table in doc.tables:
                table_lines = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    table_lines.append("| " + " | ".join(cells) + " |")
                if table_lines:
                    # 添加表头分隔线
                    if len(table_lines) > 1:
                        header = table_lines[0]
                        sep = "| " + " | ".join(["---"] * len(header.split("|")[1:-1])) + " |"
                        table_lines.insert(1, sep)
                    full_text += "\n".join(table_lines) + "\n\n"

            # 图片 OCR
            if self.enable_ocr:
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for i, shape in enumerate(doc.inline_shapes):
                            if shape.type == 3:  # 内嵌图片
                                img_path = os.path.join(temp_dir, f"image_{i}.png")
                                with open(img_path, "wb") as f:
                                    f.write(shape.image.blob)
                                img = Image.open(img_path)
                                ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng")
                                if ocr_text.strip():
                                    try:
                                        ocr_text = utils._process_text(ocr_text) if hasattr(utils,
                                                                                            '_process_text') else ocr_text.strip()
                                    except:
                                        ocr_text = ocr_text.strip()
                                    full_text += f"[图片OCR识别内容]:\n{ocr_text}\n\n"
                except Exception as e:
                    print(f"Word图片OCR异常: {e}")
        except Exception as e:
            print(f"处理Word失败: {e}")

        # 【修复点】统一清洗
        try:
            full_text = utils._process_text(full_text) if hasattr(utils, '_process_text') else full_text.strip()
        except:
            full_text = full_text.strip()
        return full_text

    def pre_detect_encoding(self, file_path: str, sample_size: int = 10240) -> str:
        """探测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(sample_size)
            result = chardet.detect(raw)
            if result['confidence'] < 0.7:
                return 'utf-8'
            return result['encoding'].lower()
        except Exception:
            return 'utf-8'

    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        encoding = self.pre_detect_encoding(file_path)
        ext_name = os.path.splitext(file_path)[1].lower()

        if ext_name == ".pdf":
            full_text = self._extract_text_from_pdf_with_ocr(file_path, encoding)
            documents = [Document(page_content=full_text, metadata={"source": file_path})]
        elif ext_name in [".docx", ".doc"]:
            full_text = self._extract_text_from_doc_with_ocr(file_path, encoding)
            documents = [Document(page_content=full_text, metadata={"source": file_path})]
        elif ext_name == ".txt":
            loader = TextLoader(file_path, encoding=encoding)
            documents = loader.load()
        else:
            raise ValueError(f"不支持的文件类型: {ext_name}")

        return self.split_documents(documents)

    def load_directory(self, directory_path: str) -> List[Document]:
        """加载目录"""
        all_documents = []
        try:
            for root, _, files in os.walk(directory_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    ext = os.path.splitext(file_path)[1].lower()

                    # 跳过非目标文件
                    if ext not in [".txt", ".pdf", ".docx", ".doc"]:
                        continue

                    encoding = self.pre_detect_encoding(file_path)
                    docs = []

                    if ext == ".txt":
                        loader = TextLoader(file_path, encoding=encoding)
                        docs = loader.load()
                    elif ext == ".pdf":
                        full_text = self._extract_text_from_pdf_with_ocr(file_path, encoding)
                        docs = [Document(page_content=full_text, metadata={"source": file_path})]
                    elif ext in [".docx", ".doc"]:
                        full_text = self._extract_text_from_doc_with_ocr(file_path, encoding)
                        docs = [Document(page_content=full_text, metadata={"source": file_path})]

                    all_documents.extend(docs)
            return self.split_documents(all_documents)
        except Exception as e:
            print(f"加载目录失败: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        if self.text_splitter is None:
            raise ValueError("切分器未初始化")

        split_docs = self.text_splitter.split_documents(documents)

        # 保存切块到本地
        self.save_chunks_to_local(split_docs)

        # 设置 Chunk ID
        for doc in split_docs:
            doc.metadata["chunk_id"] = self._global_chunk_counter
            self._global_chunk_counter += 1

        return split_docs

    def save_chunks_to_local(self, documents: List[Document], output_dir: str = "data/chunk_documents") -> None:
        """
        保存切块到本地
        注意：此处修改了默认路径以匹配 load_directory 中的逻辑，或者反之亦可，需保持一致。
        """
        if not documents:
            print("警告: 没有文档块需要保存。")
            return

        os.makedirs(output_dir, exist_ok=True)

        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", 0)
            source_path = doc.metadata.get("source", "unknown_file")
            base_name = os.path.splitext(os.path.basename(source_path))[0]

            # 生成文件名
            filename = f"chunk_{chunk_id:04d}_{base_name}.md"  # 建议使用 .md 以便查看格式
            filepath = os.path.join(output_dir, filename)

            # 构建内容 (带元数据头部)
            metadata_str = "---\n"
            for key, value in doc.metadata.items():
                # 简单的转义，防止 YAML 格式错误
                value_str = str(value).replace("\n", " ")
                metadata_str += f"{key}: {value_str}\n"
            metadata_str += "---\n\n"
            content = metadata_str + doc.page_content
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"保存文件失败 {filepath}: {e}")

    def process_documents(self, input_path: str) -> List[Document]:
        """统一入口"""
        if os.path.isfile(input_path):
            return self.load_document(input_path)
        elif os.path.isdir(input_path):
            return self.load_directory(input_path)
        else:
            raise ValueError(f"无效路径: {input_path}")


documentProcessor = DocumentProcessor()
