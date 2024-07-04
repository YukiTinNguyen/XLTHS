Bài 2: Nhận dạng nguyên âm không phụ thuộc người nói dùng đặc trưng phổ FFT

Input: tín hiệu tiếng nói (chứa 01 nguyên âm và khoảng lặng) của tập kiểm thử
(thư mục NguyenAmKiemThu-16k gồm 21 người, 105 file test).
Output: 
- Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, …,/u/), Đúng/Sai (dựa vào nhãn tên file).
- Xuất 05 vector đặc trưng biểu diễn 05 nguyên âm trên cùng 01 đồ thị.
- Bảng thống kê độ chính xác nhận dạng tổng hợp (%) theo số chiều của vector đặc 
trưng (là số điểm lấy mẫu trên miền tần số N_FFT với 03 giá trị 512, 1024, 2048).
- Ma trận nhầm lẫn (confusion matrix) của trường hợp có độ chính xác tổng hợp cao 
nhất trong 3 giá trị N_FFT trên: ma trận này thống kê số lần nhận dạng đúng/sai của 
mỗi cặp nguyên âm (có highlight nguyên âm đc nhận dạng đúng và bị nhận dạng sai 
nhiều nhất)
-------------(CÁCH LÀM)-------------------------------------------------------------------
Dữ liệu sử dụng để thiết lập và hiệu chỉnh các thông số của thuật toán: tín hiệu tiếng nói
 (mỗi tín hiệu chứa 01 nguyên âm ở giữa và 2 khoảng lặng ở 2 đầu) của tập huấn luyện
(thư mục NguyenAmHuanLuyen-16k gồm 21 người, 105 file huấn luyện).
THUẬT TOÁN:
Cài đặt BT nhận dạng theo mô hình tương tự như BT tìm kiếm âm thanh gồm 3 thuật toán sau:
1. Phân đoạn tín hiệu thành nguyên âm và khoảng lặng (slide Chapter6_SPEECH SIGNAL PROCESSING).
2. Trích xuất vector đặc trưng phổ của 05 nguyên âm dựa trên tập huấn luyện (gồm 21 người, 105 file huấn luyện):
a. Đánh dấu vùng có đặc trưng phổ ổn định đặc trưng cho nguyên âm: chia vùng chứa nguyên âm tìm được ở bước 1 thành
          3 đoạn có độ dài bằng nhau và lấy đoạn nằm giữa (giả sử gồm M khung).
b. Trích xuất vector FFT của 1 khung tín hiệu với số chiều là N_FFT (=512, 1024, 2048) dùng các hàm thư viện.
c. Tính vector đặc trưng cho 1 nguyên âm của 1 người nói = Trung bình cộng của M vector FFT của M khung thuộc vùng ổn định.
d. Tính vector đặc trưng cho 1 nguyên âm của nhiều người nói = Trung bình cộng của các vector đặc trưng cho 1 nguyên âm
        của 21 người nói (trong tập huấn luyện).
4. So khớp vector FFT của tín hiệu nguyên âm đầu vào (thuộc tập kiểm thử) với 5 vector đặc trưng đã trích xuất của 5 nguyên âm
    (dựa trên tập huấn luyện) để đưa ra kết quả nhận dạng nguyên âm bằng cách tính 5 khoảng cách Euclidean giữa 2 vector và
   đưa ra quyết định nhận dạng dựa trên k/c nhỏ nhất (hàm này SV tự cài đặt).

__________________________________________
Bài 3: Nhận dạng nguyên âm không phụ thuộc người nói dùng đặc trưng phổ MFCC

BT3 là phần mở rộng của BT2 nhằm cải thiện độ chính xác nhận dạng với mô tả các phần 
Input, Output và Yêu cầu tương tự như BT2. Điểm khác biệt ở chỗ vector đặc trưng phổ FFT 
(phản ánh nội dung chi tiết của phổ) được thay bằng vector đặc trưng phổ MFCC (phản ánh 
đường bao phổ) và thuật toán phân cụm K-mean được sử dụng để gom các người nói có 
chất giọng giống nhau vào từng cụm. Do đó các mô tả được cập nhật như sau:
Input: 
Tín hiệu tiếng nói (chứa 01 nguyên âm và 2 khoảng lặng) trong tập kiểm thử (gồm 21 người, 
105 file test).
Output: 
- Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, …,/u/), Đúng/Sai 
(dựa vào nhãn tên file).
- Xuất 05 vector đặc trưng MFCC biểu diễn 05 nguyên âm trên cùng 01 đồ thị.
- Kết quả độ chính xác nhận dạng tổng hợp (%) theo số chiều của vector đặc trưng 
N_MFCC (N_MFCC cố định là 13) và K (là số cụm với 04 giá trị K=2,3,4,5).
- Ma trận nhầm lẫn (confusion matrix) của trường hợp có độ chính xác tổng hợp cao 
nhất trong 4 giá trị K trên: ma trận này thống kê số lần nhận dạng đúng/sai của mỗi 
cặp nguyên âm (có highlight nguyên âm đc nhận dạng đúng và bị nhận dạng sai 
nhiều nhất).
Dữ liệu sử dụng để thiết lập và hiệu chỉnh các thông số của thuật toán: 
tín hiệu tiếng nói (mỗi tín hiệu chứa 01 nguyên âm ở giữa và 2 khoảng lặng ở 2 đầu) của tập 
huấn luyện (thư mục NguyenAmHuanLuyen-16k gồm 21 người, 105 file huấn luyện).
Yêu cầu:
Cài đặt BT nhận dạng theo mô hình tương tự như BT tìm kiếm âm thanh (trong TLTK [4]) 
gồm 3 thuật toán sau:
1. Phân đoạn tín hiệu thành nguyên âm và khoảng lặng (slide Chapter6_SPEECH SIGNAL 
PROCESSING).
2. Trích xuất vector đặc trưng phổ của 05 nguyên âm dựa trên tập huấn luyện (gồm 21 
người, 105 file huấn luyện):
a. Đánh dấu vùng có đặc trưng phổ ổn định đặc trưng cho nguyên âm: chia vùng 
chứa nguyên âm tìm được ở bước 1 thành 3 đoạn có độ dài bằng nhau và lấy 
đoạn nằm giữa (giả sử gồm M khung).
b. Trích xuất vector MFCC (mel-frequency cepstral coefficients) của 1 khung tín 
hiệu với số chiều (chính là số lượng hệ số MFCC) là N_MFCC =13, dùng các 
hàm của thư viện Voicebox (Matlab) hoặc libbrosa (python).
c. Tính vector đặc trưng cho 1 nguyên âm của 1 người nói = Trung bình cộng 
của M vector MFCC của M khung thuộc vùng ổn định.
d. Tính vector đặc trưng cho 1 nguyên âm của nhiều người nói = Trung bình 
cộng của các vector đặc trưng cho 1 nguyên âm của 21 người nói (trong tập 
huấn luyện).
3. So khớp vector MFCC của tín hiệu nguyên âm đầu vào (thuộc tập kiểm thử) với 5 
vector đặc trưng đã trích xuất của 5 nguyên âm (dựa trên tập huấn luyện) để đưa ra 
kết quả nhận dạng nguyên âm bằng cách tính 5 khoảng cách Euclidean giữa 2 vector 
và đưa ra quyết định nhận dạng dựa trên k/c nhỏ nhất (hàm này SV tự cài đặt).

Phần nâng cao (Kết hợp phân cụm K-means để gán nhãn):
- Mục 2c và 2d: Nếu chỉ tính 1 vector đặc trưng cho 1 nguyên âm của nhiều người nói 
thì độ chính xác biểu diễn không cao do các người nói có chất giọng ít/nhiều khác 
nhau " làm giảm độ chính xác nhận dạng. Do đó, có thể tăng độ chính xác biểu diễn 
bằng cách tính K vector đặc trưng cho 1 nguyên âm của nhiều người nói dùng thuật 
toán phân cụm K-trung bình (K-mean clustering) với K=2,3,4,5. Chạy K-mean 
clustering trên tất cả các vector MFCC của các khung nằm trong phần ổn định của 1 
nguyên âm của 21 người trong tập huấn luyện để thu được K vector trung bình làm K 
vector đặc trưng cho 1 nguyên âm.
- Mục 3: So khớp vector MFCC của tín hiệu nguyên âm đầu vào (thuộc tập kiểm thử) 
với 5xK vector đặc trưng đã trích xuất của 5 nguyên âm (dựa trên tập huấn luyện) để
đưa ra kết quả nhận dạng nguyên âm: tính 5xK khoảng cách Euclidean giữa 2 vector 
và đưa ra quyết định nhận dạng dựa trên k/c nhỏ nhất (SV tự cài đặt).
- Thuật toán phân cụm K-trung bình: SV có thể tự cài đặt hoặc dùng hàm thư viện có 
sẵn.
- Lập bảng báo cáo kết quả độ chính xác nhận dạng tổng hợp (%) theo số cụm K.
