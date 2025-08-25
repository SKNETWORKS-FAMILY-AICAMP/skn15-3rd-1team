"""
Google Drive API 연동 모듈
SKN15 폴더에서 강의록.txt 파일을 다운로드하는 기능
"""
from __future__ import annotations
import io
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class GoogleDriveClient:
    """Google Drive API 클라이언트"""
    
    def __init__(self):
        self.service = None
        self.credentials = None
        
        # OAuth2 설정 (환경변수에서 로드)
        self.CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your_client_id_here")
        self.CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "your_client_secret_here") 
        self.REDIRECT_URI = "http://localhost:8501/"  # Streamlit 기본 포트
        self.SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def _get_oauth_config(self) -> Dict[str, Any]:
        """OAuth2 설정 딕셔너리 생성"""
        return {
            "web": {
                "client_id": self.CLIENT_ID,
                "client_secret": self.CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.REDIRECT_URI]
            }
        }
    
    def authenticate(self) -> bool:
        """Google Drive 인증 수행"""
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google API 라이브러리가 설치되지 않았습니다. 'pip install google-api-python-client google-auth-oauthlib' 실행하세요.")
        
        try:
            # 저장된 토큰이 있으면 사용
            token_file = Path(".google_token.json")
            if token_file.exists():
                creds_data = json.loads(token_file.read_text())
                self.credentials = Credentials.from_authorized_user_info(creds_data, self.SCOPES)
                
                if self.credentials.valid:
                    self.service = build('drive', 'v3', credentials=self.credentials)
                    return True
            
            # OAuth2 플로우 시작
            flow = Flow.from_client_config(
                self._get_oauth_config(),
                scopes=self.SCOPES,
                redirect_uri=self.REDIRECT_URI
            )
            
            # 인증 URL 생성
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            return False, auth_url  # Streamlit에서 처리하도록 URL 반환
            
        except Exception as e:
            raise Exception(f"Google Drive 인증 실패: {str(e)}")
    
    def complete_auth(self, auth_code: str) -> bool:
        """인증 코드로 인증 완료"""
        try:
            flow = Flow.from_client_config(
                self._get_oauth_config(),
                scopes=self.SCOPES,
                redirect_uri=self.REDIRECT_URI
            )
            
            # 인증 코드로 토큰 교환
            flow.fetch_token(code=auth_code)
            
            self.credentials = flow.credentials
            self.service = build('drive', 'v3', credentials=self.credentials)
            
            # 토큰 저장
            token_file = Path(".google_token.json")
            token_file.write_text(self.credentials.to_json())
            
            return True
            
        except Exception as e:
            raise Exception(f"인증 완료 실패: {str(e)}")
    
    def find_skn15_folder(self) -> Optional[str]:
        """SKN15 폴더 ID 찾기 (대소문자 구분 없이)"""
        if not self.service:
            return None
        
        try:
            # 먼저 정확한 이름으로 검색
            results = self.service.files().list(
                q="name='SKN15' and mimeType='application/vnd.google-apps.folder'",
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            folders = results.get('files', [])
            if folders:
                return folders[0]['id']
            
            # 대소문자 구분 없이 검색
            results = self.service.files().list(
                q="mimeType='application/vnd.google-apps.folder'",
                spaces='drive',
                fields='files(id, name)',
                pageSize=100
            ).execute()
            
            all_folders = results.get('files', [])
            for folder in all_folders:
                if folder['name'].upper() == 'SKN15':
                    return folder['id']
            
            return None
            
        except Exception as e:
            raise Exception(f"SKN15 폴더 검색 실패: {str(e)}")
    
    def find_lecture_file(self, folder_id: str) -> Optional[str]:
        """SKN15 폴더 내에서 강의록.txt 파일 찾기"""
        if not self.service:
            return None
        
        try:
            results = self.service.files().list(
                q=f"name='강의록.txt' and parents in '{folder_id}'",
                spaces='drive',
                fields='files(id, name, modifiedTime)'
            ).execute()
            
            files = results.get('files', [])
            
            if files:
                return files[0]['id']  # 첫 번째 강의록.txt 파일 반환
            return None
            
        except Exception as e:
            raise Exception(f"강의록.txt 파일 검색 실패: {str(e)}")
    
    def download_file(self, file_id: str, local_path: Path) -> bool:
        """파일 다운로드"""
        if not self.service:
            return False
        
        try:
            # 파일 메타데이터 가져오기
            file_metadata = self.service.files().get(fileId=file_id).execute()
            
            # 파일 다운로드
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # 로컬 파일로 저장
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(fh.getvalue())
            
            return True
            
        except Exception as e:
            raise Exception(f"파일 다운로드 실패: {str(e)}")
    
    def download_lecture_file(self, local_path: Path = Path("강의록.txt")) -> bool:
        """SKN15/강의록.txt 파일을 다운로드하는 전체 프로세스"""
        try:
            # 1. SKN15 폴더 찾기
            folder_id = self.find_skn15_folder()
            if not folder_id:
                raise Exception("SKN15 폴더를 찾을 수 없습니다.")
            
            # 2. 강의록.txt 파일 찾기
            file_id = self.find_lecture_file(folder_id)
            if not file_id:
                raise Exception("SKN15 폴더에서 강의록.txt 파일을 찾을 수 없습니다.")
            
            # 3. 파일 다운로드
            success = self.download_file(file_id, local_path)
            
            if success:
                return True
            else:
                raise Exception("파일 다운로드에 실패했습니다.")
                
        except Exception as e:
            raise Exception(f"강의록 다운로드 실패: {str(e)}")


def is_google_drive_available() -> bool:
    """Google Drive API가 사용 가능한지 확인"""
    return GOOGLE_AVAILABLE