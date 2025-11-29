import React, { useState } from 'react';
import axios from 'axios'; 
import './App.css';

// 분석 결과를 표시하는 공통 컴포넌트
// result (AI 분석 요약 정보) / videoUrl (분석된 비디오의 url)
function AnalysisResultDisplay({ result, videoUrl }) {
  // 1. 백엔드 기본 URL (파일 서버 주소)
  const BACKEND_URL = 'http://127.0.0.1:8000';
  let faceCount = 0;

  // '내 분석 보기'로 조회 시 result는 문자열이라서 파싱해서 객체로 변환해줘야 함.
  let parsedResult = result;

  if (typeof result === 'string') {
    try {
      // 1. Python 딕셔너리 문자열 {'key': val}을 JSON 표준 {"key": val}로 바꿈
      const jsonString = result.replace(/'/g, '"');
      // 2. JSON 문자열을 실제 객체로 파싱
      parsedResult = JSON.parse(jsonString);
    } catch (e) {
      console.error("분석 결과(result) 파싱 실패:", e, "원본:", result);
      parsedResult = {}; // 실패 시 빈 객체로
    }
  }

  // 파싱된 객체(parsedResult)에서 값을 찾습니다.
  if (parsedResult && parsedResult.faces_blurred) {
    faceCount = parsedResult.faces_blurred;
  } 

return (
  <div style={{ 
      marginTop: '30px', 
      padding: '20px',
      border: '1px solid #61DAFB',
      borderRadius: '8px',
      textAlign: 'left',
      width: '80%',
      maxWidth: '640px'
  }}>
    <h4 style={{ marginTop: 0 }}>📊 분석 결과</h4>
      <p>총 {faceCount}개의 얼굴이 블러 처리되었습니다.</p>
      
      {/* 2. 블러 처리된 비디오 플레이어 */}
      {videoUrl && (
        <div>
          <p><strong>블러 처리된 영상:</strong></p>
          <video 
            controls // 재생 컨트롤러 표시
            width="100%" 
            src={`${BACKEND_URL}${videoUrl}`} // (예: http://.../static/blurred_...mp4)
            type="video/mp4"
            key={videoUrl}
          >
            브라우저가 비디오 태그를 지원하지 않습니다.
          </video>
        </div>
      )}
    </div>
  );
}

function App() {

    // 폼 데이터를 한번에 관리
    const [uploadForm, setUploadForm] = useState({
      email: '',
      password: '',
    });
    
    // 사용자가 선택한 비디오 파일을 저장할 state(값이나 속성)
    const [selectedFile, setSelectedFile] = useState(null);

    // 업로드 상태 메시지를 저장할 state
    const [uploadStatus, setUploadStatus] = useState('');

    // AI 탐지 결과를 저장할 state
    const [uploadResult, setUploadResult] = useState(null);

    const [uploadVideoUrl, setUploadVideoUrl] = useState('');

    // 분석 보기 폼을 위한 state
    const [viewForm, setViewForm] = useState({ request_id: '', password: '' });
    const [viewStatus, setViewStatus] = useState('');
    const [viewResult, setViewResult] = useState(null);
    const [viewVideoUrl, setViewVideoUrl] = useState('');

    // 분석 요청 관련 함수
    const handleUploadFormChange = (event) => {
      const{ name, value } = event.target;
      setUploadForm(prev => ({
        ...prev,
        [name]: value,
      }));
    };
  
    // 파일 선택 시 이전 결과 초기화
    const handleFileChange = (event) => {
      // 사용자가 선택한 파일 (files[0]이 첫 번째 파일)
      setSelectedFile(event.target.files[0]);
      setUploadStatus('');
      setUploadResult(null);
      setUploadVideoUrl('');
    };

    // "업로드" 버튼을 클릭할 때 이전 결과 초기화 및 새 결과 저장
    const handleUpload = () => {
      if(!selectedFile || !uploadForm.email || !uploadForm.password) {
        alert('이메일, 비밀번호, 비디오 파일을 모두 입력하세요!');
        return;
      }

      // FormData에 모든 데이터 담기
      const postData = new FormData();
      postData.append('email', uploadForm.email);
      postData.append('password', uploadForm.password);
      postData.append('video', selectedFile);

      // 분석 중 메시지 표시
      setUploadStatus('분석을 요청 중입니다......');
      setUploadResult(null);
      setUploadVideoUrl('');

      // 새 API 주소로 요청
      axios.post('http://127.0.0.1:8000/request-analysis/', postData)
        .then(response => {
          console.log("새 분석 요청 성공:", response.data);
          setUploadStatus(`성공: ${response.data.message} (ID: ${response.data.request_id})`);
        
        // 요약 정보와 "영상 URL"을 state에 저장
        setUploadResult(response.data.analysis); 
        setUploadVideoUrl(response.data.analyzed_video_url); 
      })
      .catch(error => {
        // 업로드 중 오류 발생 시
        console.error('요청 실패:', error);
        // 구체적인 오류 메시지
        const errorMsg = error.response?.data?.detail || '콘솔을 확인하세요.';
        setUploadStatus(`요청 실패: ${errorMsg}`);
      });
    };

    // 분석 보기 (View) 관련 함수
    const handleViewFormChange = (event) => {
      const { name, value } = event.target;
      setViewForm(prev => ({ ...prev, [name]: value }));
    };

    // (이 함수가 "결과 확인하기" 버튼의 onClick에 연결됩니다)
    const handleViewRequest = () => {
      if (!viewForm.request_id || !viewForm.password) {
          alert("요청 ID와 비밀번호를 모두 입력해주세요.");
          return;
      }

      setViewStatus("조회 중입니다...");
      setViewResult(null); // 기존 결과 초기화
      setViewVideoUrl(''); // 기존 비디오 URL 초기화

      // API 요청 (main.py의 /get-analysis/ 호출)
      axios.post('http://127.0.0.1:8000/get-analysis/', {
          request_id: viewForm.request_id,
          password: viewForm.password
      })
      .then(response => {
          // 성공 시 (main.py에서 db_post를 반환)
          const data = response.data;
          console.log("조회 성공:", data);
          setViewStatus(`요청 ID ${data.id} 조회 성공`);

          // DB에서 받은 영상 경로와 분석 결과를 state에 저장
          setViewResult(data.analysis_result); // (예: "{'faces_blurred': 10}")
          setViewVideoUrl(data.analyzed_video_path); // (예: "/static/blurred_...mp4")
      })
      .catch(error => {
          console.error("조회 에러:", error);
          const errorMsg = error.response?.data?.detail || "조회 중 오류가 발생했습니다.";
          setViewStatus(`조회 실패: ${errorMsg}`);
      });
    };

        

    // --- 렌더링 UI ---
  return (
    <div className="App">
      <header className="App-header">

        {/* --- 1. 분석 요청 폼  --- */}
        <h3>새 분석 요청 (얼굴 블러)</h3>
        <div style={{ margin: '10px' }}>
          <label style={{ marginRight: '10px' }}>이메일:</label>
          <input type="email" name="email" value={uploadForm.email} onChange={handleUploadFormChange} />
        </div>
        <div style={{ margin: '10px' }}>
          <label style={{ marginRight: '10px' }}>비밀번호:</label>
          <input type="password" name="password" value={uploadForm.password} onChange={handleUploadFormChange} />
        </div>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ margin: '10px' }}
        />
        <button onClick={handleUpload} style={{ marginTop: '10px', fontSize: '16px' }}>
          블러 요청하기
        </button>
        {uploadStatus && (
          <p style={{ marginTop: '20px', color: '#61DAFB' }}>{uploadStatus}</p>
        )}
        {/* 업로드 성공 시 결과 표시 */}
        {uploadResult && <AnalysisResultDisplay result={uploadResult} videoUrl={uploadVideoUrl} />}

        {/* --- 구분선 --- */}
        <hr style={{ width: '80%', margin: '40px 0' }} />

        {/* --- 2. 분석 보기 폼 --- */}
        <h3>내 분석 보기</h3>
        
        {/* --- "내 분석 보기"의 JSX를 viewForm 상태와 연결 --- */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
            <div>
                <label>요청 ID: </label>
                <input 
                    type="text" 
                    name="request_id" // state 키와 매칭
                    value={viewForm.request_id} 
                    onChange={handleViewFormChange} 
                    placeholder="ID 입력"
                />
            </div>
            <div>
                <label>비밀번호: </label>
                <input 
                    type="password" 
                    name="password" // state 키와 매칭
                    value={viewForm.password} 
                    onChange={handleViewFormChange} 
                    placeholder="비밀번호 입력"
                />
            </div>
            {/* onClick에 handleViewRequest 함수 연결 */}
            <button onClick={handleViewRequest}>결과 확인하기</button>
        </div>

        {/* 조회 상태 메시지 (viewStatus 사용) */}
        <p style={{ color: '#61DAFB' }}>{viewStatus}</p>

        {/* --- "내 분석 보기"의 결과 표시 --- */}
        {/* (조회 성공 시 viewResult와 viewVideoUrl을 사용해 AnalysisResultDisplay 컴포넌트 재사용) */}
        {viewVideoUrl && <AnalysisResultDisplay result={viewResult} videoUrl={viewVideoUrl} />}

      </header>
    </div>
  );
}

export default App;