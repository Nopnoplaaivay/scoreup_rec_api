import os
import numpy as np

from pymongo import MongoClient, DESCENDING, ASCENDING
from src.utils.logger import LOGGER


class MongoDBManager:
    """Khỏi tạo course_id Nhập môn Công nghệ thông tin"""

    def __init__(self, database_name="codelab1"):
        # self.client = MongoClient(os.getenv("LOCAL_MONGO_URL"))
        self.client = MongoClient(os.getenv("MONGO_URL"))
        self.db = self.client[database_name]

    def check_connection(self):
        try:
            databases = self.client.list_database_names()
            LOGGER.info("Connection successful!")
            LOGGER.info(f"Databases: {databases}")
        except Exception as e:
            LOGGER.error(f"Connection failed: {e}")


    def get_questions_by_course(self, course_id="c3a788eb31f1471f9734157e9516f9b6"):
        """Lấy tất cả các câu hỏi trong một khóa học"""
        return self.db["questions"].find({"notionDatabaseId": course_id}, sort=[("chapter", ASCENDING)])
    
    def get_knowledge_concepts_by_course(self):
        """Lấy tất cả các khái niệm"""
        return self.db["knowledge_concepts"].find()