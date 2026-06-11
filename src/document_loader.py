import os
import tempfile
import platform
import pdfplumber
import pytesseract
from PIL import Image
from typing import List
from src.config import config
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
        # 调用本类内部方法获取poppler_path
        self.poppler_path = self._get_poppler_path()
        # 调用本类内部方法获取tesseract_path
        self.tesseract_path = self._setup_tesseract_path()

    def _setup_tesseract_path(self):
        """智能配置 Tesseract OCR 引擎路径"""
        if platform.system() != "Windows":
            return  # Linux/Mac 依赖系统 PATH，通常不需要手动配置
        # 1. 优先从你的项目配置文件或环境变量读取（最灵活）
        env_path = getattr(config, 'TESSERACT_BIN_PATH', None)
        if env_path and os.path.exists(env_path):
            pytesseract.pytesseract.tesseract_cmd = env_path
            print(f"从配置加载 Tesseract 路径: {env_path}")
            return
        # 2. 尝试常见的默认安装路径
        default_paths = [
            r"E:\damoxing\python-project\help_ocr\tesseract\install\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        for path in default_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"自动发现 Tesseract 路径: {path}")
                return

        print("警告: 未在 Windows 上找到 Tesseract，OCR 功能将无法使用！")

    def _get_poppler_path(self):
        """
        智能获取 Poppler 路径：
        1. 如果是 Linux/Mac，直接返回 None (依赖系统 PATH)
        2. 如果是 Windows，优先读取环境变量，否则尝试常见默认路径
        """
        if platform.system() != "Windows":
            return None
        # 1. 尝试从环境变量读取（最灵活，推荐在 .env 文件中配置）
        env_path = config.POPPLER_BIN_PATH
        if env_path and os.path.exists(env_path):
            return env_path
            # 2. 尝试常见的默认安装路径 (根据你的实际解压位置修改)
        default_paths = [
            r"E:\damoxing\python-project\help_ocr\poppler-26.02.0\Library\bin",
            r"C:\Program Files\poppler\Library\bin",
            r".\poppler\Library\bin"
        ]
        for path in default_paths:
            if os.path.exists(os.path.join(path, "pdftoppm.exe")):
                print(f"自动发现 Poppler 路径: {path}")
                return path

        print(" 警告: 未在 Windows 上找到 Poppler 路径，PDF 扫描件处理可能会失败。")
        return None

    def _extract_text_from_pdf_with_ocr(self, file_path: str) -> str:
        """
        智能提取 PDF 文本：优先提取原生文本，若文本极少（判定为扫描件），则整页 OCR。
        :param file_path: pdf 文件路径
        :return: 提取文本的内容
        """
        full_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                # 尝试提取前几页的原生文本，判断是否为扫描件
                sample_text = ""
                for page in pdf.pages[:3]:
                    text = page.extract_text()
                    if text:
                        sample_text += text

                is_scanned = len(sample_text.strip()) < 50  # 阈值设定：如果前3页提取字符少于50个，视为扫描件

                # 【第二步】根据判断结果选择处理方式
                if not is_scanned:
                    # 正常文档型 PDF：逐页提取文本 + 局部图片OCR
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n\n"

                        if self.enable_ocr:
                            try:
                                for img in page.images:
                                    x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
                                    img_crop = page.crop((x0, top, x1, bottom))
                                    img_obj = img_crop.to_image()
                                    ocr_text = pytesseract.image_to_string(img_obj.original, lang="chi_sim+eng")
                                    if ocr_text.strip():
                                        full_text += f"[图片OCR识别内容]:\n{ocr_text}\n\n"
                            except Exception:
                                pass
                else:
                    # 扫描件 PDF：使用 pdf2image 将整页转为图片后全量 OCR
                    if not self.enable_ocr:
                        raise ValueError("当前PDF为扫描件且未开启OCR功能，无法提取文本。")

                    print(f"检测到扫描件格式，正在将 {file_path} 转换为图片并进行OCR...")
                    # dpi=200 保证识别清晰度
                    pages = convert_from_path(
                        file_path,
                        dpi=200,
                        poppler_path=self.poppler_path)
                    for page_num, page_img in enumerate(pages, 1):
                        try:
                            ocr_text = pytesseract.image_to_string(page_img, lang="chi_sim+eng")
                            if ocr_text.strip():
                                full_text += f"=== 第{page_num}页 ===\n{ocr_text}\n\n"
                        except Exception as e:
                            print(f"第{page_num}页OCR失败: {e}")

        except Exception as e:
            raise ValueError(f"处理pdf文件失败:{str(e)}")
        return full_text

    def _extract_text_from_doc_with_ocr(self, file_path: str) -> str:
        """
        使用 python-docx 提取 Word 文本，并对内嵌图片进行 OCR 识别
        """
        full_text = ""
        try:
            doc = DocxDocument(file_path)
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text += paragraph.text + "\n\n"
            # 处理word表格内容为markdown
            for table in doc.tables:
                table_lines = []
                for row_idx, row in enumerate(table.rows):
                    cells = [cell.text.strip() for cell in row.cells]
                    table_lines.append("| " + " | ".join(cells) + " |")
                    if row_idx == 0:
                        table_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                if table_lines:
                    full_text += "\n".join(table_lines) + "\n\n"
            # 开启ocr扫描处理
            if self.enable_ocr:
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for i, shape in enumerate(doc.inline_shapes):
                            if shape.type == 3:
                                img_path = os.path.join(temp_dir, f"image_{i}.png")
                                image = shape.image
                                # 写入文件
                                with open(img_path, "wb") as f:
                                    f.write(image.blob)
                                img = Image.open(img_path)
                                ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng")
                                if ocr_text.strip():
                                    full_text += f"[图片OCR识别内容]:\n{ocr_text}\n\n"
                except Exception:
                    pass
        except Exception as e:
            raise ValueError(f"处理Word文件失败: {str(e)}")
        return full_text

    # 如果是文件就直接切分分文件
    def load_document(self, file_path: str) -> List[Document]:
        """根据文件路径，读取并切割单个文档"""
        ext_name = os.path.splitext(file_path)[1].lower()
        if ext_name == ".pdf":
            full_text = self._extract_text_from_pdf_with_ocr(file_path)
            documents = [Document(page_content=full_text, metadata={"source": file_path})]
        elif ext_name in [".docx", ".doc"]:
            full_text = self._extract_text_from_doc_with_ocr(file_path)
            documents = [Document(page_content=full_text, metadata={"source": file_path})]
        elif ext_name == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext_name}")

        return self.split_documents(documents)

    # 如果是文件夹，递归遍历所有文件切割
    def load_directory(self, directory_path: str) -> List[Document]:
        """递归读取目录下所有支持的文档"""
        all_documents = []
        try:
            # root _ file 当前文件夹路径，子文件夹（用不到_表示） 文件列表
            for root, _, files in os.walk(directory_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    # 通过splitext获取文件扩展名
                    ext = os.path.splitext(file_path)[1].lower()
                    docs = []
                    if ext == ".txt":
                        loader = TextLoader(file_path, encoding='utf-8')
                        docs = loader.load()
                    elif ext == ".pdf":
                        full_text = self._extract_text_from_pdf_with_ocr(file_path)
                        docs = [Document(page_content=full_text, metadata={"source": file_path})]
                    elif ext in [".docx", ".doc"]:
                        full_text = self._extract_text_from_doc_with_ocr(file_path)
                        docs = [Document(page_content=full_text, metadata={"source": file_path})]
                    else:
                        continue
                    all_documents.extend(docs)
            return self.split_documents(all_documents)
        except Exception as e:
            print(f"Error loading directory: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将多个文档内容切割成带有元数据的小文本块"""
        if self.text_splitter is None:
            raise ValueError("Text splitter is not initialized.")
        split_docs = self.text_splitter.split_documents(documents)
        for doc in split_docs:
            doc.metadata["chunk_id"] = self._global_chunk_counter
            self._global_chunk_counter += 1
        return split_docs

    def process_documents(self, input_path: str) -> List[Document]:
        """统一的文档处理入口方法"""
        if os.path.isfile(input_path):
            docs = self.load_document(input_path)
        elif os.path.isdir(input_path):
            docs = self.load_directory(input_path)
        else:
            raise ValueError(f"Invalid path: {input_path}")
        return docs


def save_chunks_to_files(processor: DocumentProcessor, input_path: str, output_dir: str = "debug_chunks"):
    """
    将处理后的分块保存为文本文件，方便肉眼排查
    """
    print(f"正在将文档 {input_path} 切分为 chunks 并保存到 {output_dir} ...")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 调用处理器获取分块
    chunks = processor.process_documents(input_path)

    # 清空目录下的旧文件（可选）
    # for f in os.listdir(output_dir): os.remove(os.path.join(output_dir, f))

    # 保存每个分块为单独的文件
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.metadata.get("chunk_id", i)
        source = os.path.basename(chunk.metadata.get("source", "unknown"))

        # 文件名： chunk_0001_source_xxx.txt
        filename = f"chunk_{chunk_id:04d}_{source}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Source: {source} ===\n")
            f.write(f"=== Chunk ID: {chunk_id} ===\n")
            f.write(f"=== Content Length: {len(chunk.page_content)} ===\n\n")
            f.write(chunk.page_content)

    print(f" 完成分块保存！共保存了 {len(chunks)} 个分块。")
    return chunks


documentProcessor = DocumentProcessor()
