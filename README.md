# AI_PROJECT_SERVER
AI 프로젝트 포스텍 서버에 올라갈 서버부분 구현
포스텍 서버에 putty로 접속 후, git clone을 통해 모두 다운받고 pip install -r requirements.txt로 모든 패키지 설치하세요.
터미널 창에서 python app.py로 실행하게 되면 각 포스텍 주소 141.223.140.xx:8787 포트로 외부에서 접속 시 엘리베이터 화면을 확인할 수 있음.
def gen()이라고 구현되어 있는 부분에, emit('message' ~ ) 로 구현되어 있는 부분이 실제 인물을 탐지하게 되면, 인물정보와 층수를 전송하는 부분.
현재 ../users 데이터에는 이름_floor_company 순으로 저장되어 있음(ex. sangrae_25_posco) 각 회사에 맞는 키워드를 사전에 설정해놓으면
기사 요약 서비스를 제공할 때 company name == posco이면 keyword = '포스코 채용' 이런식으로 네이버 api로 기사 크롤링 후 요약 제공하면 좋을 듯.

video = cv2.VideoCapture('http://poscopro ~)라고 되어 있는 부분이 있는데,
자기 컴퓨터를 webcam으로 사용하고 싶다면 AI_PROJECT의 stream.py를 자기 노트북에서 실행하되 ngrok이라는 프로그램 다운받아서
윈도우 터미널(명령프롬프트)에서 ngrok이 있는 폴더에 접근 후 'ngrok http 8787' 과 같은 식으로 입력하면 자기 ip + port 번호를 자동으로 주소로 지정해줌
저렇게 명령어를 입력하면 fowarding에 http ~ 로 시작하는 주소가 있을 것임. 그 주소를 복사해서 cv2.VideoCapture('여기' + '/video_feed')에 여기부분에 넣으면
해당 주소에서 웹캠 비디오를 읽어옴을 확인할 수 있음 (반드시 뒤에 + '/video_feed')를 입력할것.

