import React, { useState, useEffect, useRef } from 'react'; 
import axios from 'axios'; 

const BACKEND_URL = 'http://127.0.0.1:8000'; 

// 🚨 상태 텍스트 변환 헬퍼 함수 (영문 -> 한글)
const getStatusText = (status) => {
    if (status === 'PENDING') return '민원 처리 대기중';
    if (status === 'IN_PROGRESS') return '민원 처리중';
    if (status === 'COMPLETED') return '민원 처리 완료';
    return status; // 매칭되는 게 없으면 그대로 표시
};

// --- PasswordPrompt (동일) ---
function PasswordPrompt({ onConfirm, onBack, mode = 'view' }) {
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleSubmit = () => {
        if (!password.trim()) {
            setError('비밀번호를 입력해주세요.');
            return;
        }
        onConfirm(password);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') handleSubmit();
    };

    const titleText = mode === 'delete' ? '글 삭제 확인' : '비밀번호 확인';
    const buttonText = mode === 'delete' ? '삭제하기' : '확인';
    const labelText = mode === 'delete' ? '삭제하려면 비밀번호를 입력하세요.' : '게시글 비밀번호를 입력하세요.';

    return (
        <div style={styles.adminAuthWrapper}>
            <header style={styles.header}>
                <h1 style={styles.headerTitle}>{titleText}</h1>
                <button onClick={onBack} style={styles.backButton}>뒤로가기</button>
            </header>
            <main style={styles.main}>
                <div style={styles.authBox}>
                    <p style={styles.authLabel}>{labelText}</p>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => {
                            setPassword(e.target.value);
                            setError('');
                        }}
                        onKeyPress={handleKeyPress}
                        style={styles.authInput}
                        placeholder="비밀번호 입력"
                    />
                    {error && <p style={styles.authError}>{error}</p>}
                    <button 
                        onClick={handleSubmit} 
                        style={mode === 'delete' ? styles.deleteConfirmButton : styles.authButton}
                    >
                        {buttonText}
                    </button>
                </div>
            </main>
        </div>
    );
}

// --- AnalysisResultPage (동일) ---
function AnalysisResultPage({ post, onBack, currentViewName }) {
    const backHandler = currentViewName.startsWith('analysis_result_admin') ? onBack.admin : onBack.list;

    if (!post || post.status !== 'COMPLETED' || !post.analyzed_video_path) {
        return (
            <div style={styles.detailWrapper}>
                <button onClick={backHandler} style={styles.detailBackButton}>뒤로가기</button>
                <p style={styles.statusCell}>영상 분석 중입니다.</p>
            </div>
        );
    }
    
    // 🚨 [수정] 결과가 여러 개일 수 있으므로 처리
    let videoPaths = [];
    try {
        const parsed = JSON.parse(post.analyzed_video_path);
        if (Array.isArray(parsed)) videoPaths = parsed;
        else videoPaths = [post.analyzed_video_path];
    } catch (e) {
        videoPaths = [post.analyzed_video_path]; // JSON 파싱 실패 시 단일 문자열로 취급
    }

    return (
        <div style={styles.detailWrapper}>
            <header style={styles.detailHeader}>
                <h1 style={styles.detailTitle}>민원 처리 결과: {post.title}</h1>
                <button onClick={backHandler} style={styles.detailBackButton}>
                    뒤로가기
                </button>
            </header>

            <main style={styles.resultMain}>
                <div style={styles.resultInfoBox}>
                     <div style={{margin: '15px 0', color: '#2e7d32'}}>
                        <p style={styles.statusCompleted}>✅ 민원 처리 완료</p>
                        <p style={{fontSize: '14px', color: '#555'}}>
                            총 {videoPaths.length}개의 영상 분석이 완료되었습니다.<br/>
                            얼굴(타원형)과 번호판(직사각형)이 비식별화되었습니다.
                        </p>
                    </div>
                </div>

                {/* 🚨 [수정] 여러 영상 반복 렌더링 */}
                {videoPaths.map((path, idx) => {
                    const actualFileName = path.substring(path.lastIndexOf('/') + 1); 
                    const videoUrl = `${BACKEND_URL}${path}`; 
                    const downloadApiUrl = `${BACKEND_URL}/api/download/${encodeURIComponent(actualFileName)}`;
                    
                    return (
                        <div key={idx} style={styles.videoCard}>
                            <h4 style={{marginBottom: '10px'}}>영상 #{idx + 1}</h4>
                            <div style={styles.videoPlayerContainer}>
                                <video controls width="100%" src={videoUrl} type="video/mp4">
                                    브라우저가 비디오 태그를 지원하지 않습니다.
                                </video>
                            </div>
                            <a href={downloadApiUrl} download={actualFileName} style={styles.downloadButton}>
                                📥 영상 다운로드 ({actualFileName})
                            </a>
                        </div>
                    );
                })}
            </main>
        </div>
    );
}

// --- PostDetail (다중 업로드 기능 수정됨) ---
function PostDetail({ postId, onBack, onAnalyze, currentViewName, goToAnalysisResult, onDeletePost, onEditPost }) {
  const [post, setPost] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false); 
  
  // 🚨 [수정] 단일 파일(adminFile) -> 다중 파일 배열(adminFiles)
  const [adminFiles, setAdminFiles] = useState([]);
  const fileInputRef = useRef(null);

  const isAdminView = currentViewName === 'admin_detail'; 
  const backHandler = currentViewName === 'admin_detail' ? onBack.admin : onBack.list;

  const fetchDetail = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`http://127.0.0.1:8000/api/posts/${postId}`);
        if (!response.ok) throw new Error('정보를 불러오는 데 실패했습니다.');
        const data = await response.json();
        setPost(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

  useEffect(() => {
    fetchDetail();
  }, [postId, isAnalyzing]); 

  const handleStopAnalysis = async () => {
      if (!window.confirm("⚠️ 경고: 현재 진행 중인 분석을 강제로 중지하시겠습니까?")) return;
      
      try {
          await axios.post(`${BACKEND_URL}/admin/stop/${post.id}`);
          alert("분석 중지 요청을 보냈습니다. 잠시 후 상태가 변경됩니다.");
          fetchDetail(); 
      } catch (e) {
          alert(`중지 요청 실패: ${e.message}`);
      }
  };

  if (isLoading || isAnalyzing) {
    return (
      <div style={styles.detailWrapper}>
        <button onClick={backHandler} style={styles.detailBackButton}>뒤로가기</button>
        <p style={styles.statusCell}>
            {isAnalyzing ? '민원 처리(영상 업로드 및 AI 분석) 진행 중...' : '로딩 중...'}
        </p>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div style={styles.detailWrapper}>
        <button onClick={backHandler} style={styles.detailBackButton}>뒤로가기</button>
        <p style={styles.statusCell}>오류: {error || '게시글을 찾을 수 없습니다.'}</p>
      </div>
    );
  }
  
  const formattedDate = new Date(post.created_at).toLocaleString();

  // 🚨 [수정] 파일 선택 핸들러 (버그 수정됨)
  const handleFileChange = (e) => {
      const files = e.target.files; 
      if (files && files.length > 0) {
          const newFiles = Array.from(files);
          setAdminFiles(prev => [...prev, ...newFiles]);
      }
      e.target.value = '';
  };

  // 🚨 [신규] 파일 목록에서 제거
  const handleRemoveFile = (index) => {
      setAdminFiles(prev => prev.filter((_, i) => i !== index));
  };

  // 🚨 [수정] 분석 시작 핸들러 (다중 파일)
  const handleAnalyzeClick = async () => {
      if (adminFiles.length === 0) {
          alert("분석할 영상 파일을 하나 이상 추가해주세요.");
          return;
      }

      if (!window.confirm(`총 ${adminFiles.length}개의 파일로 분석을 시작하시겠습니까?`)) return;

      setIsAnalyzing(true); 
      try {
          const analyzedPost = await onAnalyze(post.id, adminFiles); 
          goToAnalysisResult(analyzedPost); 
      } catch (e) {
          alert(`처리 실패: ${e.message}`);
      } finally {
          setIsAnalyzing(false);
      }
  };

  const handleViewResultClick = () => {
      goToAnalysisResult(post); 
  };
  
  const handleDeleteClick = () => {
      if (window.confirm("정말로 삭제하시겠습니까?")) onDeletePost(post.id); 
  }
  
  const handleEditClick = () => {
      onEditPost(post);
  }

  return (
    <div style={styles.detailWrapper}>
      <header style={styles.detailHeader}>
        <h1 style={styles.detailTitle}>{post.title}</h1>
        <button onClick={backHandler} style={styles.detailBackButton}>뒤로가기</button>
      </header>
      
      <div style={styles.detailMeta}>
        <div style={styles.detailMetaInfo}>
          <p style={styles.detailMetaText}>작성자: {post.author} ({post.email})</p>
          <p style={styles.detailMetaText}>작성일: {formattedDate}</p>
        </div>
      </div>
      
      <div style={styles.detailStatusContainer}>
        <p style={styles.detailStatusLabel}>처리 현황</p>
        <div style={styles.statusBadgeContainer}>
            {post.status === 'COMPLETED' ? (
                post.analyzed_video_path && (
                    <button onClick={handleViewResultClick} style={styles.viewResultButtonInline}>
                        ▶ 분석 영상 보기
                    </button>
                )
            ) : (
                <span style={post.status === 'IN_PROGRESS' ? {...styles.statusPendingLarge, color: '#ffc107'} : styles.statusPendingLarge}>
                    {getStatusText(post.status)}
                </span>
            )}
        </div>
      </div>

      <div style={styles.detailSection}>
        <h3>민원 내용</h3>
        <div style={styles.detailContentBox}>
          {post.content || '내용 없음'}
        </div>
      </div>
      
      <div style={styles.detailSection}>
        <h3 style={styles.fileHeader}>
          요청 정보 / 관리
          {!isAdminView && ( 
              <div style={styles.buttonGroup}>
                  <button onClick={handleEditClick} style={styles.editButton}>✏️ 글 수정</button>
                  <button onClick={handleDeleteClick} style={styles.deleteButton}>🗑️ 글 삭제</button>
              </div>
          )}
        </h3>
        <p><strong>영상이 필요한 곳(주소):</strong> {post.target_address}</p>
        {post.original_video_filename && (
            <p style={{fontSize:'14px', color:'#666', marginTop:'5px'}}>
                (관리자 확보 영상: {post.original_video_filename})
            </p>
        )}
      </div>

      {isAdminView && post.status === 'PENDING' && (
          <div style={styles.adminUploadBox}>
              {/* 숨겨진 파일 입력 (multiple 허용) */}
              <input 
                  type="file" 
                  accept="video/*" 
                  multiple
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  style={{display: 'none'}}
              />
              
              <div style={{width: '100%'}}>
                  <p style={{fontWeight: 'bold', marginBottom: '10px'}}>분석할 영상 목록 ({adminFiles.length}개)</p>
                  
                  {/* 파일 목록 표시 */}
                  {adminFiles.length > 0 ? (
                      <div style={styles.fileListContainer}>
                          {adminFiles.map((file, idx) => (
                              <div key={idx} style={styles.fileItem}>
                                  {/* 🚨 [수정] 이모티콘 제거 및 번호 표시 */}
                                  <span style={styles.fileName}>{idx + 1}. {file.name}</span>
                                  {/* 🚨 [수정] X 버튼 -> 삭제 버튼 텍스트 변경 */}
                                  <button onClick={() => handleRemoveFile(idx)} style={styles.removeFileButton}>삭제</button>
                              </div>
                          ))}
                      </div>
                  ) : (
                      <div style={styles.emptyFileBox}>
                          <p style={{color: '#999'}}>추가된 영상이 없습니다.</p>
                      </div>
                  )}

                  {/* 버튼 그룹 */}
                  <div style={{display: 'flex', gap: '10px', marginTop: '15px', justifyContent: 'center'}}>
                      <button 
                          onClick={() => fileInputRef.current.click()} 
                          style={styles.addFileButton}
                      >
                          + 영상 추가하기
                      </button>
                      <button 
                          onClick={handleAnalyzeClick} 
                          style={adminFiles.length > 0 ? styles.adminAnalyzeButton : styles.disabledButton}
                          disabled={adminFiles.length === 0}
                      >
                          민원 처리 시작 ({adminFiles.length}개)
                      </button>
                  </div>
              </div>
          </div>
      )}

      {isAdminView && post.status === 'IN_PROGRESS' && (
          <div style={styles.adminStopContainer}>
              <p style={{...styles.statusPendingLarge, color: '#856404', margin: '0 0 10px 0'}}>
                  ⏳ 현재 민원 처리(AI 분석) 중입니다...
              </p>
              <p style={{fontSize: '14px', color: '#666', marginBottom: '15px'}}>
                  완료될 때까지 기다려 주세요. 문제가 생겼다면 아래 버튼을 눌러 중지할 수 있습니다.
              </p>
              <button onClick={handleStopAnalysis} style={styles.stopButton}>
                  🛑 분석 강제 중지
              </button>
          </div>
      )}
    </div>
  );
}

// --- PostEditForm (동일) ---
function PostEditForm({ post, onBack, onUpdateComplete }) {
    const [title, setTitle] = useState(post.title);
    const [content, setContent] = useState(post.content);
    const [address, setAddress] = useState(post.target_address);
    const [password, setPassword] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleSubmit = async () => {
        if (!title || !password || !address) {
            alert('제목, 주소, 비밀번호는 필수입니다.');
            return;
        }

        if (isSubmitting) return;
        setIsSubmitting(true);

        try {
            const response = await axios.put(`${BACKEND_URL}/api/posts/${post.id}`, {
                title: title,
                content: content,
                target_address: address,
                password: password 
            });

            if (response.status === 200) {
                alert("성공적으로 수정되었습니다.");
                onUpdateComplete(post.id); 
            }
        } catch (error) {
            console.error("Update Error:", error);
            const msg = error.response?.data?.detail || "수정에 실패했습니다.";
            alert(`오류: ${msg}`);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div style={styles.detailWrapper}>
            <header style={styles.header}>
                <h1 style={styles.headerTitle}>글 수정하기</h1>
                <button onClick={onBack} style={styles.backButton}>취소</button>
            </header>

            <main style={styles.main}>
                <div style={styles.formGroup}>
                    <label style={styles.formLabel} htmlFor="edit-title">글 제목</label>
                    <input
                        type="text"
                        id="edit-title"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        style={styles.formInput}
                    />
                </div>

                <div style={styles.formGroup}>
                    <label style={styles.formLabel} htmlFor="edit-address">영상이 필요한 곳 (주소)</label>
                    <input
                        type="text"
                        id="edit-address"
                        value={address}
                        onChange={(e) => setAddress(e.target.value)}
                        style={styles.formInput}
                    />
                </div>

                <div style={styles.formGroup}>
                    <label style={styles.formLabel} htmlFor="edit-content">민원 내용</label>
                    <textarea
                        id="edit-content"
                        value={content}
                        onChange={(e) => setContent(e.target.value)}
                        style={{...styles.formInput, ...styles.formTextarea}} 
                    />
                </div>

                <div style={{...styles.formGroup, borderTop: '1px solid #eee', paddingTop: '20px', marginTop: '20px'}}>
                    <label style={{...styles.formLabel, color: '#dc3545'}} htmlFor="edit-password">비밀번호 확인 (필수)</label>
                    <input
                        type="password"
                        id="edit-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        style={styles.formInput}
                        placeholder="글 작성 시 설정한 비밀번호를 입력하세요"
                    />
                    <p style={{fontSize: '13px', color: '#666', marginTop: '5px'}}>
                        정보를 안전하게 보호하기 위해 비밀번호를 입력해야 수정이 완료됩니다.
                    </p>
                </div>

                <div style={styles.formActions}>
                    <button onClick={handleSubmit} style={styles.submitButton} disabled={isSubmitting}>
                        {isSubmitting ? '저장 중...' : '수정 완료'}
                    </button>
                </div>
            </main>
        </div>
    );
}

// --- AnalysisForm (개인정보 동의 추가) ---
function AnalysisForm({ onBack }) {
  const [title, setTitle] = useState('');
  const [password, setPassword] = useState('');
  const [author, setAuthor] = useState(''); 
  const [email, setEmail] = useState(''); 
  const [content, setContent] = useState('');
  const [address, setAddress] = useState(''); 
  const [isSubmitting, setIsSubmitting] = useState(false); 
  
  // 🚨 [신규] 개인정보 동의 체크 상태
  const [agreePrivacy, setAgreePrivacy] = useState(false);
  const [agreeService, setAgreeService] = useState(false);
  const [showPrivacyDetail, setShowPrivacyDetail] = useState(false);
  const [showServiceDetail, setShowServiceDetail] = useState(false);
  
  const backHandler = onBack.list; 

  const handleSubmit = async () => {
    if (!title || !author || !password || !email || !address) {
      alert('제목, 작성자, 비밀번호, 이메일, 주소는 필수입니다.');
      return;
    }
    // 🚨 [신규] 동의 체크 확인
    if (!agreePrivacy || !agreeService) {
      alert('필수 동의 항목을 모두 체크해주세요.');
      return;
    }
    if (isSubmitting) return; 
    
    setIsSubmitting(true);

    const formData = new FormData();
    formData.append('title', title);
    formData.append('author', author);
    formData.append('content', content);
    formData.append('email', email);
    formData.append('password', password);
    formData.append('target_address', address); 

    try {
      const response = await axios.post(`${BACKEND_URL}/request-analysis/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.status === 200) {
        alert(`민원 접수가 완료되었습니다.`);
        backHandler(); 
      }
    } catch (error) {
      console.error("Form Submission Error:", error);
      alert(`오류 발생: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsSubmitting(false); 
    }
  };

  return (
    <>
      <header style={styles.header}>
        <h1 style={styles.headerTitle}>민원 접수 글쓰기</h1>
        <button onClick={backHandler} style={styles.backButton}>
          뒤로가기
        </button>
      </header>

      <main style={styles.main}>
        <div style={styles.formGroup}>
          <label style={styles.formLabel} htmlFor="title">글 제목</label>
          <input
            type="text"
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            style={styles.formInput}
            placeholder="제목을 입력하세요"
          />
        </div>

        <div style={styles.formGroupFlex}> 
          <div style={styles.formGroupHalf}> 
            <label style={styles.formLabel} htmlFor="password">비밀번호</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={styles.formInput}
              placeholder="글 비밀번호를 입력하세요"
            />
          </div>
          <div style={styles.formGroupHalf}>
            <label style={styles.formLabel} htmlFor="author">작성자 이름</label>
            <input
              type="text"
              id="author"
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              style={styles.formInput}
              placeholder="작성자 이름을 입력하세요"
            />
          </div>
        </div>

        <div style={styles.formGroup}>
          <label style={styles.formLabel} htmlFor="email">이메일 주소</label>
          <input
            type="email" 
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={styles.formInput}
            placeholder="결과를 통보받을 이메일"
          />
        </div>

        <div style={styles.formGroup}>
          <label style={styles.formLabel} htmlFor="address">영상이 필요한 곳 (주소)</label>
          <input
            type="text"
            id="address"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            style={styles.formInput}
            placeholder="예: 대구광역시 북구 산격동 1234"
          />
        </div>

        <div style={styles.formGroup}>
          <label style={styles.formLabel} htmlFor="content">민원 내용</label>
          <textarea
            id="content"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            style={{...styles.formInput, ...styles.formTextarea}} 
            placeholder="민원 내용을 상세히 적어주세요."
          />
        </div>

        {/* 🚨 [신규] 개인정보 동의 섹션 */}
        <div style={styles.agreementSection}>
          <h3 style={styles.agreementTitle}>📋 필수 동의 항목</h3>
          
          {/* 개인정보 수집·이용 동의 */}
          <div style={styles.agreementItem}>
            <label style={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={agreePrivacy}
                onChange={(e) => setAgreePrivacy(e.target.checked)}
                style={styles.checkbox}
              />
              <span style={styles.checkboxText}>[필수] 개인정보 수집·이용에 동의합니다.</span>
            </label>
            <button 
              type="button"
              onClick={() => setShowPrivacyDetail(!showPrivacyDetail)}
              style={styles.detailToggleButton}
            >
              {showPrivacyDetail ? '접기' : '상세보기'}
            </button>
          </div>
          
          {showPrivacyDetail && (
            <div style={styles.agreementDetailBox}>
              <p style={styles.agreementDetailTitle}>「개인정보보호법」 제15조에 따른 개인정보 수집·이용 동의</p>
              <ul style={styles.agreementList}>
                <li><strong>수집 항목:</strong> 이름, 이메일, 해당 요청 주소</li>
                <li><strong>수집·이용 목적:</strong> 영상 비식별화 처리 서비스 제공 및 결과 전달</li>
                <li><strong>보유 기간:</strong> 30일간 보관 후 자동 삭제 및 사용자 요청 시 즉시 삭제</li>
                <li><strong>동의 거부권:</strong> 동의를 거부할 수 있으나, 거부 시 서비스 이용이 제한됩니다.</li>
              </ul>
            </div>
          )}

          {/* 서비스 이용 안내 동의 */}
          <div style={styles.agreementItem}>
            <label style={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={agreeService}
                onChange={(e) => setAgreeService(e.target.checked)}
                style={styles.checkbox}
              />
              <span style={styles.checkboxText}>[필수] 서비스 이용 안내를 확인하였습니다.</span>
            </label>
            <button 
              type="button"
              onClick={() => setShowServiceDetail(!showServiceDetail)}
              style={styles.detailToggleButton}
            >
              {showServiceDetail ? '접기' : '상세보기'}
            </button>
          </div>
          
          {showServiceDetail && (
            <div style={styles.agreementDetailBox}>
              <p style={styles.agreementDetailTitle}>서비스 이용 안내</p>
              <ul style={styles.agreementList}>
                <li><strong>서비스 내용:</strong> 요청하신 주소 인근 CCTV를 수집 후, 다소 민감한 개인정보(얼굴, 차량번호판)를 AI 기반으로 비식별화(블러) 처리합니다.</li>
                <li><strong>처리 절차:</strong> 민원 접수 → 담당자 영상 확보 → AI 분석 → 결과 안내</li>
                <li><strong>유의사항:</strong> AI 자동 처리 특성상 일부 개인정보 비식별화 처리가 누락될 수 있으며, 처리된 영상은 30일간 보관 후 자동 삭제됩니다.</li>
              </ul>
            </div>
          )}
        </div>

        <div style={styles.formActions}>
          <button 
            onClick={handleSubmit} 
            style={agreePrivacy && agreeService ? styles.submitButton : styles.disabledSubmitButton} 
            disabled={isSubmitting || !agreePrivacy || !agreeService}
          >
            {isSubmitting ? '접수 중...' : '민원 접수'}
          </button>
        </div>
      </main>
    </>
  );
}

// --- AdminAuth (동일) ---
function AdminAuth({ onBack, onAuthSuccess }) {
    const ADMIN_PASS = "1234"; 
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleAuth = () => {
        if (password === ADMIN_PASS) {
            onAuthSuccess();
        } else {
            setError('비밀번호가 일치하지 않습니다.');
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') handleAuth();
    };
    
    const backHandler = onBack.list;

    return (
        <div style={styles.adminAuthWrapper}>
            <header style={styles.header}>
                <h1 style={styles.headerTitle}>관리자 인증</h1>
                <button onClick={backHandler} style={styles.backButton}>뒤로가기</button>
            </header>
            <main style={styles.main}>
                <div style={styles.authBox}>
                    <p style={styles.authLabel}>관리자 비밀번호 입력</p>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        onKeyPress={handleKeyPress}
                        style={styles.authInput}
                    />
                    {error && <p style={styles.authError}>{error}</p>}
                    <button onClick={handleAuth} style={styles.authButton}>인증</button>
                </div>
            </main>
        </div>
    );
}

// --- AdminPanel (동일) ---
function AdminPanel({ onBack, goToDetailView, onDeletePost }) {
    const [posts, setPosts] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [totalPostsCount, setTotalPostsCount] = useState(0); 
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [currentTab, setCurrentTab] = useState('all');

    const pagesToShow = 5;
    const backHandler = onBack.list;

    const getStatusFilter = (tab) => {
        if (tab === 'pending') return 'PENDING'; 
        if (tab === 'completed') return 'COMPLETED';
        return '';
    };

    const refreshPosts = async () => {
        setIsLoading(true);
        setError(null);

        try {
            const statusFilter = getStatusFilter(currentTab);
            const url = new URL('http://127.0.0.1:8000/api/posts');
            url.searchParams.append('page', currentPage);
            url.searchParams.append('search', searchTerm);
            if (statusFilter) {
                url.searchParams.append('status_filter', statusFilter);
            }

            const response = await fetch(url.toString());
            if (!response.ok) throw new Error('데이터 로드 실패');
            const data = await response.json();
            
            setPosts(data.posts); 
            setTotalPages(data.total_pages);
            setTotalPostsCount(data.total_posts); 

        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        refreshPosts();
    }, [currentPage, searchTerm, currentTab]);

    const handleAdminDelete = (postId) => {
        if (window.confirm("관리자 권한으로 삭제하시겠습니까?")) {
            onDeletePost(postId).then(() => refreshPosts());
        }
    };

    const renderPagination = () => {
        const currentPageBlock = Math.ceil(currentPage / pagesToShow);
        const startPage = (currentPageBlock - 1) * pagesToShow + 1;
        const endPage = Math.min(startPage + pagesToShow - 1, totalPages);
        const pageNumbers = [];
        for (let i = startPage; i <= endPage; i++) { if (i > 0) pageNumbers.push(i); }
        const handleSearchChange = (e) => { setSearchTerm(e.target.value); setCurrentPage(1); };
        return { pageNumbers, startPage, endPage, handleSearchChange };
    };

    const { pageNumbers, startPage, endPage, handleSearchChange } = renderPagination();

    const handleTabClick = (tabName) => {
        setCurrentTab(tabName);
        setCurrentPage(1);
    };

    return (
        <>
            <header style={styles.header}>
                <h1 style={styles.headerTitle}>관리자 페이지</h1>
                <button onClick={backHandler} style={styles.backButton}>로그아웃</button>
            </header>

            <main style={styles.main}>
                <div style={styles.adminTabs}>
                    <button style={currentTab === 'all' ? styles.adminTabActive : styles.adminTab} onClick={() => handleTabClick('all')}>전체 목록</button>
                    <button style={currentTab === 'pending' ? styles.adminTabActive : styles.adminTab} onClick={() => handleTabClick('pending')}>미처리 목록</button>
                    <button style={currentTab === 'completed' ? styles.adminTabActive : styles.adminTab} onClick={() => handleTabClick('completed')}>처리 완료 목록</button>
                </div>

                <div style={styles.boardHeader}>
                    <h2 style={styles.boardTitle}>민원 목록</h2>
                    <div style={styles.searchContainer}>
                        <input type="text" placeholder="검색" style={styles.searchInput} value={searchTerm} onChange={handleSearchChange} />
                        <button style={styles.searchButton}>검색</button>
                    </div>
                </div>
                
                <table style={styles.table}> 
                    <thead>
                        <tr>
                            <th style={styles.tableHeader}>번호</th>
                            <th style={styles.tableHeader}>제목</th>
                            <th style={styles.tableHeader}>작성자</th>
                            <th style={styles.tableHeader}>접수일</th>
                            <th style={styles.tableHeader}>처리 현황</th> 
                            <th style={styles.tableHeader}>관리</th>
                        </tr>
                    </thead>
                    <tbody>
                        {isLoading ? (
                            <tr><td colSpan="6" style={styles.statusCell}>로딩 중...</td></tr>
                        ) : error ? (
                            <tr><td colSpan="6" style={styles.statusCell}>오류: {error}</td></tr>
                        ) : posts.length === 0 ? (
                            <tr><td colSpan="6" style={styles.statusCell}>민원이 없습니다.</td></tr>
                        ) : (
                            posts.map((post, index) => {
                                const reverseNumber = totalPostsCount - ((currentPage - 1) * 10) - index;
                                return (
                                <tr key={post.id}>
                                    <td style={styles.tableCell}>{reverseNumber}</td>
                                    <td style={{...styles.tableCell, ...styles.linkCell}} onClick={() => goToDetailView(post.id)}>{post.title}</td>
                                    <td style={styles.tableCell}>{post.author}</td>
                                    <td style={styles.tableCell}>{new Date(post.created_at).toLocaleDateString()}</td>
                                    <td style={styles.tableCell}>
                                        <span style={post.status === 'COMPLETED' ? styles.statusCompleted : post.status === 'IN_PROGRESS' ? {...styles.statusPending, color: '#ffc107'} : styles.statusPending}>
                                            {getStatusText(post.status)}
                                        </span>
                                    </td> 
                                    <td style={styles.tableCell}>
                                        <button onClick={(e) => { e.stopPropagation(); handleAdminDelete(post.id); }} style={styles.smallDeleteButton}>삭제</button>
                                    </td>
                                </tr>
                                );
                            })
                        )}
                    </tbody>
                </table> 
                <div style={styles.paginationContainer}>
                    <button style={startPage === 1 ? styles.disabledPageButton : styles.pageButton} onClick={() => setCurrentPage(startPage - 1)} disabled={startPage === 1}>&laquo;</button>
                    {pageNumbers.map(number => (
                        <button key={number} style={number === currentPage ? styles.activePageButton : styles.pageButton} onClick={() => setCurrentPage(number)}>{number}</button>
                    ))}
                    <button style={endPage === totalPages || totalPages === 0 ? styles.disabledPageButton : styles.pageButton} onClick={() => setCurrentPage(endPage + 1)} disabled={endPage === totalPages || totalPages === 0}>&raquo;</button>
                </div>
            </main>
        </>
    );
}

// --- App ---
function App() {
  const [posts, setPosts] = useState([]);
  const [view, setView] = useState({ name: 'list', postId: null, postData: null }); 
  const [searchTerm, setSearchTerm] = useState(''); 
  const [currentPage, setCurrentPage] = useState(1); 
  const [totalPages, setTotalPages] = useState(1); 
  const [totalPostsCount, setTotalPostsCount] = useState(0); 
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null); 

  const pagesToShow = 5;

  const backHandlers = {
      list: () => setView({ name: 'list', postId: null, postData: null }),
      admin: () => setView({ name: 'admin_panel', postId: null, postData: null }),
      detail: (postId) => setView({ name: 'detail', postId: postId, postData: null })
  };

  const goToListView = backHandlers.list;
  const goToAdminPanel = backHandlers.admin;

  const goToAdminAuth = () => { setView({ name: 'admin_auth', postId: null, postData: null }); };
  const goToDetailView = (postId) => { setView({ name: 'password_check', postId: postId, postData: null }); };
  const goToAdminDetailView = (postId) => { setView({ name: 'admin_detail', postId: postId, postData: null }); };
  const handlePasswordVerified = (postId) => { setView({ name: 'detail', postId: postId, postData: null }); };

  const checkPostPassword = async (password) => {
      try {
          const response = await axios.post(`${BACKEND_URL}/api/posts/${view.postId}/verify`, { password });
          if (response.status === 200) handlePasswordVerified(view.postId);
      } catch (err) { alert('비밀번호가 일치하지 않습니다.'); }
  };
  
  const goToDeleteCheck = (postId) => { setView({ name: 'delete_check', postId: postId, postData: null }); };

  const handleDeletePost = async (postId) => {
      try {
          const response = await axios.delete(`${BACKEND_URL}/api/posts/${postId}`);
          if (response.status === 200) {
              alert('글이 삭제되었습니다.');
              if (view.name === 'admin_panel' || view.name === 'admin_detail') goToAdminPanel();
              else goToListView(); 
              return true; 
          }
      } catch (err) {
          alert(err.response?.data?.detail || '삭제에 실패했습니다.');
          return false;
      }
  };
  
  const goToAnalysisResult = (post) => {
      const viewName = view.name.startsWith('admin') ? 'analysis_result_admin' : 'analysis_result_list';
      setView({ name: viewName, postId: post.id, postData: post });
  };
  
  const goToEditPost = (post) => {
      setView({ name: 'edit_post', postId: post.id, postData: post });
  };
  
  const handleUpdateComplete = (postId) => {
      setView({ name: 'detail', postId: postId, postData: null });
  };

  // 🚨 [수정] 파일 배열 처리
  const handleAnalyze = async (postId, files) => {
      try {
          const formData = new FormData();
          // 여러 파일 append (videos 이름으로)
          files.forEach(file => {
              formData.append('videos', file);
          });

          const response = await axios.post(`${BACKEND_URL}/admin/analyze/${postId}`, formData, {
              headers: { 'Content-Type': 'multipart/form-data' }, 
          });
          
          if (response.status !== 200) throw new Error(response.data?.detail || '처리 실패');
          alert(`처리 성공: 민원 상태가 변경되었습니다.`);
          return response.data; 
      } catch (error) {
          console.error("Analysis Error:", error);
          throw error;
      }
  };

  useEffect(() => {
    const fetchPosts = async () => {
      if (view.name !== 'list') return; 
      setIsLoading(true); 
      setError(null);
      try {
        const url = new URL('http://127.0.0.1:8000/api/posts');
        url.searchParams.append('page', currentPage);
        url.searchParams.append('search', searchTerm);
        const response = await fetch(url.toString());
        if (!response.ok) throw new Error('데이터 로드 실패');
        const data = await response.json();
        setPosts(data.posts); 
        setTotalPages(data.total_pages);
        setTotalPostsCount(data.total_posts); 
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false); 
      }
    };
    fetchPosts();
  }, [currentPage, searchTerm, view.name]); 

  const renderPagination = () => {
    const currentPageBlock = Math.ceil(currentPage / pagesToShow);
    const startPage = (currentPageBlock - 1) * pagesToShow + 1;
    const endPage = Math.min(startPage + pagesToShow - 1, totalPages);
    const pageNumbers = [];
    for (let i = startPage; i <= endPage; i++) { if (i > 0) pageNumbers.push(i); }
    const handleSearchChange = (e) => { setSearchTerm(e.target.value); setCurrentPage(1); };
    return { pageNumbers, startPage, endPage, handleSearchChange };
  };

  const { pageNumbers, startPage, endPage, handleSearchChange } = renderPagination();

  return (
    <div style={styles.pageWrapper}>
      <div style={styles.container}>
        {view.name === 'list' && (
          <>
            <header style={styles.listHeader}> 
                <h1 style={styles.siteTitle}>영상 분석 민원 사이트</h1>
            </header>
            <main style={styles.mainContent}>
              <div style={styles.boardHeader}>
                <h2 style={styles.boardTitle}>민원 게시판</h2>
                <div style={styles.searchContainer}>
                  <input type="text" placeholder="검색" style={styles.searchInput} value={searchTerm} onChange={handleSearchChange} />
                  <button style={styles.searchButton}>검색</button>
                </div>
              </div>
              <table style={styles.table}> 
                <thead>
                  <tr>
                    <th style={styles.tableHeader}>번호</th>
                    <th style={styles.tableHeader}>제목</th>
                    <th style={styles.tableHeader}>작성자</th>
                    <th style={styles.tableHeader}>접수일</th>
                    <th style={styles.tableHeader}>처리 현황</th> 
                  </tr>
                </thead>
                <tbody>
                  {isLoading ? (
                    <tr><td colSpan="5" style={styles.statusCell}>로딩 중...</td></tr>
                  ) : error ? (
                    <tr><td colSpan="5" style={styles.statusCell}>오류: {error}</td></tr>
                  ) : posts.length === 0 ? (
                    <tr><td colSpan="5" style={styles.statusCell}>등록된 민원이 없습니다.</td></tr>
                  ) : (
                    posts.map((post, index) => {
                        const reverseNumber = totalPostsCount - ((currentPage - 1) * 10) - index;
                        return (
                      <tr key={post.id}>
                        <td style={styles.tableCell}>{reverseNumber}</td>
                        <td style={{...styles.tableCell, ...styles.linkCell}} onClick={() => goToDetailView(post.id)}>{post.title}</td>
                        <td style={styles.tableCell}>{post.author}</td>
                        <td style={styles.tableCell}>{new Date(post.created_at).toLocaleDateString()}</td>
                        <td style={styles.tableCell}>
                          <span style={post.status === 'COMPLETED' ? styles.statusCompleted : post.status === 'IN_PROGRESS' ? {...styles.statusPending, color: '#ffc107'} : styles.statusPending}>{getStatusText(post.status)}</span>
                        </td> 
                      </tr>
                    )})
                  )}
                </tbody>
              </table> 
              <div style={styles.paginationContainer}>
                <button style={startPage === 1 ? styles.disabledPageButton : styles.pageButton} onClick={() => setCurrentPage(startPage - 1)} disabled={startPage === 1}>&laquo;</button>
                {pageNumbers.map(number => (
                  <button key={number} style={number === currentPage ? styles.activePageButton : styles.pageButton} onClick={() => setCurrentPage(number)}>{number}</button>
                ))}
                <button style={endPage === totalPages || totalPages === 0 ? styles.disabledPageButton : styles.pageButton} onClick={() => setCurrentPage(endPage + 1)} disabled={endPage === totalPages || totalPages === 0}>&raquo;</button>
              </div>
            </main>
            <button style={styles.floatingButton} onClick={() => setView({ name: 'form', postId: null, postData: null })}>민원<br />접수</button>
            <button style={styles.adminFloatingButton} onClick={goToAdminAuth}>관리자<br />바로가기</button>
          </>
        )}
        
        {view.name === 'form' && <AnalysisForm onBack={backHandlers} />}
        {view.name === 'password_check' && <PasswordPrompt onConfirm={checkPostPassword} onBack={goToListView} mode="view" />}
        {view.name === 'delete_check' && <PasswordPrompt onConfirm={handleDeletePost} onBack={() => setView({ name: 'detail', postId: view.postId, postData: null })} mode="delete" />}
        
        {view.name === 'detail' && view.postId !== null && <PostDetail postId={view.postId} onBack={backHandlers} onAnalyze={handleAnalyze} currentViewName={view.name} goToAnalysisResult={goToAnalysisResult} onDeletePost={handleDeletePost} onEditPost={goToEditPost} />}
        
        {view.name === 'edit_post' && view.postData !== null && <PostEditForm post={view.postData} onBack={() => backHandlers.detail(view.postId)} onUpdateComplete={handleUpdateComplete} />}

        {view.name === 'admin_auth' && <AdminAuth onBack={backHandlers} onAuthSuccess={goToAdminPanel} />}
        {view.name === 'admin_panel' && <AdminPanel onBack={backHandlers} goToDetailView={goToAdminDetailView} onDeletePost={handleDeletePost} />}
        {view.name === 'admin_detail' && view.postId !== null && <PostDetail postId={view.postId} onBack={backHandlers} onAnalyze={handleAnalyze} currentViewName={view.name} goToAnalysisResult={goToAnalysisResult} onDeletePost={handleDeletePost} />}
        {(view.name === 'analysis_result_admin' || view.name === 'analysis_result_list') && view.postData !== null && <AnalysisResultPage post={view.postData} onBack={backHandlers} currentViewName={view.name} />}
      </div> 
    </div> 
  );
}

const styles = {
  // ... 기존 스타일 유지 ...
  pageWrapper: {
    backgroundColor: '#e9e9e9', 
    minHeight: '100vh',
    padding: '20px 0', 
    fontFamily: 'Arial, sans-serif',
  },
  container: {
    maxWidth: '1200px', 
    margin: '0 auto', 
    fontFamily: 'Arial, sans-serif',
    backgroundColor: 'white', 
    borderRadius: '8px', 
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)', 
    padding: '20px',
    position: 'relative', 
  },
  listHeader: {
    display: 'flex',
    justifyContent: 'center', 
    alignItems: 'center',
    padding: '10px 20px',
    borderBottom: '2px solid #f0f0f0',
    position: 'relative', 
  },
  siteTitle: {
    fontSize: '28px',
    fontWeight: 'bold',
    color: '#222', 
    margin: 0,
  },
  mainContent: {
    padding: '20px 0',
  },
  boardTitle: {
    fontSize: '24px',
    color: '#333',
  },
  boardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '15px', 
  },
  searchContainer: {
    display: 'flex',
    alignItems: 'center',
  },
  searchInput: {
    width: '250px', 
    padding: '8px 12px',
    fontSize: '15px',
    border: '1px solid #ccc',
    borderRadius: '5px',
    marginRight: '8px', 
  },
  searchButton: {
    backgroundColor: '#007BFF', 
    color: '#ffffff',
    border: 'none',
    borderRadius: '5px',
    padding: '8px 15px',
    fontSize: '15px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse', 
  },
  tableHeader: {
    backgroundColor: '#f9f9f9',
    padding: '12px',
    borderBottom: '1px solid #ddd',
    textAlign: 'left',
  },
  tableCell: {
    padding: '12px',
    borderBottom: '1px solid #eee',
  },
  linkCell: {
    cursor: 'pointer',
    color: '#007BFF',
    fontWeight: '500',
    '&:hover': {
        textDecoration: 'underline',
    },
  },
  statusCell: {
    padding: '40px 12px',
    textAlign: 'center',
    color: '#777',
    fontSize: '16px',
    borderBottom: '1px solid #eee',
  },
  statusPending: {
    color: '#dc3545', 
    fontWeight: 'bold',
  },
  statusCompleted: {
    color: '#28a745', 
    fontWeight: 'bold',
  },
  paginationContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: '25px',
  },
  pageButton: {
    padding: '8px 12px',
    margin: '0 5px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    cursor: 'pointer',
    backgroundColor: 'white',
    color: '#007BFF',
    fontSize: '15px',
    transition: 'background-color 0.2s',
  },
  activePageButton: {
    padding: '8px 12px',
    margin: '0 5px',
    border: '1px solid #007BFF',
    borderRadius: '4px',
    cursor: 'default',
    backgroundColor: '#007BFF',
    color: 'white',
    fontSize: '15px',
  },
  disabledPageButton: {
    padding: '8px 12px',
    margin: '0 5px',
    border: '1px solid #eee',
    borderRadius: '4px',
    cursor: 'not-allowed',
    backgroundColor: '#f9f9f9',
    color: '#aaa',
    fontSize: '15px',
  },
  floatingButton: {
    position: 'fixed', 
    right: '40px',
    bottom: '40px',
    width: '80px',
    height: '80px',
    backgroundColor: '#007BFF',
    color: 'white',
    border: 'none',
    borderRadius: '50%', 
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
    lineHeight: '1.3', 
    zIndex: 1000,
  },
  adminFloatingButton: {
    position: 'fixed', left: '40px', bottom: '40px', width: '80px', height: '80px', backgroundColor: '#343a40', color: 'white', border: 'none', borderRadius: '50%', cursor: 'pointer', fontSize: '16px', fontWeight: 'bold', boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)', lineHeight: '1.3', zIndex: 1000 },
  detailWrapper: {
    padding: '20px',
    backgroundColor: 'white',
  },
  detailHeader: {
    display: 'flex',                  
    justifyContent: 'space-between',  
    alignItems: 'center',
    borderBottom: '2px solid #f0f0f0',
    paddingBottom: '10px',
    marginBottom: '10px', 
  },
  detailTitle: {
    fontSize: '32px',
    fontWeight: 'bold',
    color: '#222',
    margin: '10px 0',
    flexGrow: 1, 
  },
  detailBackButton: {
    backgroundColor: '#6c757d',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    padding: '8px 15px',
    fontSize: '15px',
    fontWeight: '500',
    cursor: 'pointer',
  },
  detailMeta: {
    borderBottom: '1px solid #ddd', 
    marginBottom: '15px',
    paddingBottom: '5px',
  },
  detailMetaInfo: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  detailMetaText: {
    fontSize: '14px', 
    color: '#666',
    margin: '5px 0', 
  },
  detailStatusContainer: {
    textAlign: 'center',
    padding: '15px',
    backgroundColor: '#f8f9fa',
    borderRadius: '5px',
    marginBottom: '20px',
    border: '1px solid #e9ecef',
  },
  detailStatusLabel: {
    fontSize: '14px',
    color: '#343a40',
    fontWeight: 'bold',
    margin: '0 0 5px 0',
  },
  statusBadgeContainer: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      gap: '15px', 
  },
  statusPendingLarge: {
    color: '#dc3545', 
    fontWeight: 'bold',
    fontSize: '20px',
  },
  viewResultButtonInline: {
    backgroundColor: '#28a745', 
    color: 'white',
    border: 'none',
    borderRadius: '20px',
    padding: '5px 15px',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  },
  detailSection: {
    marginBottom: '30px',
  },
  detailContentBox: {
    minHeight: '150px',
    padding: '15px',
    border: '1px solid #ddd',
    borderRadius: '5px',
    backgroundColor: '#fff',
    whiteSpace: 'pre-wrap', 
    lineHeight: '1.6',
  },
  fileHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#333',
    marginBottom: '10px',
  },
  // 🚨 [신규] 버튼 그룹 스타일
  buttonGroup: {
      display: 'flex',
      gap: '10px',
  },
  deleteButton: {
    backgroundColor: '#dc3545', 
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    padding: '8px 15px',
    fontSize: '14px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  // 🚨 [수정] 수정 버튼 스타일 (노란색으로 변경)
  editButton: {
    backgroundColor: '#ffc107', // 노란색
    color: '#212529', // 검정색 텍스트
    border: 'none',
    borderRadius: '5px',
    padding: '8px 15px',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  // 🚨 [신규] 분석 중지 버튼 스타일
  stopButton: {
    backgroundColor: '#343a40', // 짙은 회색/검정 계열
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    padding: '10px 20px',
    fontSize: '15px',
    fontWeight: 'bold',
    cursor: 'pointer',
    marginTop: '10px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
  },
  // 🚨 [신규] 관리자 분석 중지 컨테이너
  adminStopContainer: {
      textAlign: 'center',
      padding: '20px',
      backgroundColor: '#fff3cd', // 연한 노란색 경고 배경
      borderRadius: '8px',
      border: '1px solid #ffeeba',
      marginTop: '30px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '5px'
  },
  smallDeleteButton: {
    backgroundColor: '#dc3545',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    padding: '5px 10px',
    fontSize: '12px',
    fontWeight: 'bold',
    cursor: 'pointer',
  },
  deleteConfirmButton: {
    backgroundColor: '#dc3545',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    padding: '10px 20px',
    fontSize: '16px',
    cursor: 'pointer',
  },
  adminActionContainer: {
    textAlign: 'center',
    marginTop: '30px',
  },
  // 🚨 [수정] 관리자 업로드 박스 스타일 변경 (세로 배치)
  adminUploadBox: {
      display: 'flex', 
      flexDirection: 'column', // 세로 배치로 변경
      alignItems: 'center', 
      justifyContent: 'center',
      padding: '20px',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
      border: '1px solid #e9ecef',
      marginTop: '30px'
  },
  // 🚨 [신규] 파일 목록 컨테이너
  fileListContainer: {
      width: '100%',
      maxHeight: '200px',
      overflowY: 'auto',
      border: '1px solid #ddd',
      borderRadius: '5px',
      backgroundColor: '#fff',
      padding: '10px',
      marginBottom: '15px',
      boxSizing: 'border-box', // 🚨 [수정] 박스 사이즈 초과 방지
  },
  // 🚨 [신규] 개별 파일 아이템
  fileItem: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '8px',
      borderBottom: '1px solid #eee',
  },
  // 🚨 [신규] 파일 이름
  fileName: {
      fontSize: '14px',
      color: '#333',
  },
  // 🚨 [신규] 파일 제거 버튼
  removeFileButton: {
      backgroundColor: 'transparent',
      border: 'none',
      color: '#dc3545',
      fontWeight: 'bold',
      cursor: 'pointer',
      fontSize: '14px',
  },
  // 🚨 [신규] 빈 파일 박스
  emptyFileBox: {
      width: '100%',
      padding: '20px',
      textAlign: 'center',
      border: '1px dashed #ccc',
      borderRadius: '5px',
      marginBottom: '15px',
      backgroundColor: '#fff',
      boxSizing: 'border-box', // 🚨 [수정] 박스 사이즈 초과 방지
  },
  // 🚨 [신규] 파일 추가 버튼
  addFileButton: {
      backgroundColor: '#17a2b8',
      color: 'white',
      border: 'none',
      borderRadius: '5px',
      padding: '12px 20px',
      fontSize: '16px',
      fontWeight: 'bold',
      cursor: 'pointer',
      transition: 'background-color 0.2s',
  },
  adminAnalyzeButton: {
    backgroundColor: '#dc3545',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    padding: '12px 30px',
    fontSize: '17px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  disabledButton: {
    backgroundColor: '#ccc',
    color: '#666',
    border: 'none',
    borderRadius: '5px',
    padding: '12px 30px',
    fontSize: '17px',
    fontWeight: 'bold',
    cursor: 'not-allowed',
  },
  resultMain: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    paddingTop: '20px',
  },
  // 🚨 [신규] 결과 영상 카드 (다중 결과용)
  videoCard: {
      width: '100%',
      maxWidth: '800px',
      marginBottom: '40px',
      paddingBottom: '20px',
      borderBottom: '1px solid #eee',
  },
  videoPlayerContainer: {
    width: '100%',
    marginBottom: '15px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  resultInfoBox: {
    textAlign: 'center',
    padding: '20px',
    border: '1px solid #28a745',
    backgroundColor: '#f1f8f1',
    borderRadius: '8px',
    marginBottom: '30px',
    width: '100%',
    maxWidth: '600px',
  },
  downloadButton: {
    display: 'inline-block',
    backgroundColor: '#28a745',
    color: 'white',
    textDecoration: 'none',
    padding: '10px 20px',
    borderRadius: '5px',
    marginTop: '5px',
    fontSize: '15px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '15px 0',
    borderBottom: '1px solid #e8e8e8',
  },
  headerTitle: {
    fontSize: '22px',
    fontWeight: '600',
    color: '#333',
    margin: 0,
  },
  main: {
    padding: '25px 0',
  },
  formGroup: {
    marginBottom: '25px',
  },
  formGroupFlex: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '20px',
    marginBottom: '25px',
  },
  formGroupHalf: {
    flex: 1,
  },
  formLabel: {
    display: 'block',
    marginBottom: '8px',
    fontSize: '15px',
    fontWeight: '600',
    color: '#333',
  },
  formInput: {
    width: '100%',
    padding: '12px',
    fontSize: '16px',
    border: '1px solid #ccc',
    borderRadius: '5px',
    boxSizing: 'border-box',
  },
  formTextarea: {
    minHeight: '180px',
    resize: 'vertical',
    fontFamily: 'Arial, sans-serif',
  },
  formActions: {
    textAlign: 'center', 
    marginTop: '30px',
  },
  submitButton: {
    backgroundColor: '#007BFF',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    padding: '12px 30px',
    fontSize: '17px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  backButton: {
    backgroundColor: '#6c757d',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    padding: '8px 15px',
    fontSize: '15px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  adminAuthWrapper: {
    padding: '20px',
  },
  authBox: {
    maxWidth: '400px',
    margin: '50px auto',
    padding: '30px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    backgroundColor: '#f9f9f9',
    textAlign: 'center',
  },
  authLabel: {
    fontSize: '18px',
    marginBottom: '15px',
    fontWeight: '600',
  },
  authInput: {
    width: '90%',
    padding: '10px',
    fontSize: '16px',
    border: '1px solid #ccc',
    borderRadius: '5px',
    marginBottom: '15px',
    textAlign: 'center',
  },
  authButton: {
    backgroundColor: '#007BFF',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    padding: '10px 20px',
    fontSize: '16px',
    cursor: 'pointer',
  },
  authError: {
    color: '#dc3545',
    marginBottom: '10px',
  },
  adminTabs: {
    display: 'flex',
    borderBottom: '2px solid #ddd',
    marginBottom: '20px',
  },
  adminTab: {
    padding: '10px 20px',
    fontSize: '16px',
    border: 'none',
    backgroundColor: 'transparent',
    cursor: 'pointer',
    borderBottom: '3px solid transparent',
    transition: 'all 0.2s',
    marginRight: '10px',
  },
  adminTabActive: {
    padding: '10px 20px',
    fontSize: '16px',
    border: 'none',
    backgroundColor: 'transparent',
    cursor: 'default',
    borderBottom: '3px solid #007BFF',
    color: '#007BFF',
    fontWeight: 'bold',
    marginRight: '10px',
  },
  // 🚨 [신규] 개인정보 동의 섹션 스타일
  agreementSection: {
    marginTop: '30px',
    padding: '20px',
    backgroundColor: '#f8f9fa',
    borderRadius: '8px',
    border: '1px solid #dee2e6',
  },
  agreementTitle: {
    fontSize: '16px',
    fontWeight: 'bold',
    marginBottom: '15px',
    color: '#333',
  },
  agreementItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 0',
    borderBottom: '1px solid #e9ecef',
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
  },
  checkbox: {
    width: '18px',
    height: '18px',
    marginRight: '10px',
    cursor: 'pointer',
  },
  checkboxText: {
    fontSize: '14px',
    color: '#333',
  },
  detailToggleButton: {
    backgroundColor: 'transparent',
    border: '1px solid #6c757d',
    borderRadius: '4px',
    padding: '4px 10px',
    fontSize: '12px',
    color: '#6c757d',
    cursor: 'pointer',
  },
  agreementDetailBox: {
    backgroundColor: '#fff',
    border: '1px solid #dee2e6',
    borderRadius: '6px',
    padding: '15px',
    marginTop: '10px',
    marginBottom: '10px',
  },
  agreementDetailTitle: {
    fontSize: '14px',
    fontWeight: 'bold',
    marginBottom: '10px',
    color: '#495057',
  },
  agreementList: {
    margin: 0,
    paddingLeft: '20px',
    fontSize: '13px',
    lineHeight: '1.8',
    color: '#555',
  },
  disabledSubmitButton: {
    backgroundColor: '#ccc',
    color: '#666',
    border: 'none',
    borderRadius: '5px',
    padding: '12px 30px',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'not-allowed',
  },
};

export default App;
