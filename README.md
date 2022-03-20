# Graduation-Project

<h3> 1. Check Seat : 학생이 자리에 있는지 없는지 확인 </h3>
 - 과정 01 : 학생이 고개를 갸웃거리고 있을 때 인식이 잘 안되는 부분을 보완하기 위해 캡쳐한 프레임을 왼쪽으로 90도, 오른쪽으로 90도까지 10도씩 사진을 회전시켜 얼굴을 인식 <br>
 - 과정 02 : 얼굴 인식이 될 경우 학생이 자리에 앉아있는 것으로 판단하여 다음 알고리즘으로 넘어간다. <br>
 - 과정 03 : 얼굴 인식이 안될 경우 학생이 자리에 앉아있지 않은 것으로 판단 <br>

<h3> 2. Tilted Head : 고개를 갸웃거리는지 확인 </h3>
 - 과정 01 : Mediapipe Pose과 MobileNet 방식을 결합 <br>
 - 과정 02 : Mediapiep Pose의 경우 코와 입 중점, 어깨 왼쪽과 오른쪽을 이은 두 직선의 사이각을 계산하여 일정 각도 이하로 나타나면 고개를 갸웃거리는 것으로 판단 <br>
 - 과정 03 : MobileNet의 경우, 얼굴의 XYZ축 각도를 계산하여 특정 조건을 만족할 경우 고개를 갸웃거리는 것으로 판단 <br>
 - 과정 04 : Mediapipe Pose 방식을 이용하기 위해서는 Shoulder Landmark가 필요 <br>
 - 과정 05 : 만약 Shoulder Landmark가 인식될 경우, Mediapipe Pose 방식을 이용해 알고리즘 결과 측정 <br>
 - 과정 06 : 만약 Shoulder Landmark가 인식되지 않을 경우, Mediapipe Pose 방식을 이용해 알고리즘 결과 측정 <br>
   
<table>
    <tr>
        <td> Shoulder Landmark Check Result </td>
        <td> Mediapipe Pose Result </td>
        <td> MobileNet Result </td>
        <td> Tilted Head Result </td>
    </tr>
    <tr>
        <td rowspan="4"> O </td>
        <td rowspan="2"> O </td>
        <td> - </td>
        <td rowspan="2"> O </td>
    </tr>
    <tr>
        <td> - </td>
    </tr>
    <tr>
        <td rowspan="2"> X </td>
        <td> - </td>
        <td rowspan="2"> X </td>
    </tr>
    <tr>
        <td> - </td>
    </tr>
    <tr>
        <td rowspan="2"> X </td>
        <td> - </td>
        <td> O </td>
        <td> O</td>
    </tr>
    <tr>
        <td> - </td>
        <td> X </td>
        <td> X </td>
    </tr>
</table>

<h3> 3. Frown Grabella : 미간을 찌푸리고 있는지 확인 </h3>
 - 방식 :  <br>
 - 결과 :  <br>

<h3> 4. Face Closer : 얼굴을 가까이했는지 확인 </h3>
 - 과정 01 : 정해진 초마다 한번씩 프레임을 캡쳐 <br>
 - 과정 02 : 캡쳐한 프레임에 나타나는 왼쪽 귀와 오른쪽 귀 사이의 거리를 계산 <br>
 - 과정 03 : 최근 프레임 10개를 기준으로 귀 사이 거리 평균을 구하고 이를 이용하여 귀 사이 거리 변화값이 기준값보다 크다면 얼굴을 가까이한 것으로 판단 <br>

<h3> 5. Eye Tracking : 홍채의 움직임을 인식 </h3>
 - 과정 :  <br>
 - 결과 :  <br>

# < 알고리즘 작동 순서 >
1. Check Seat <br>
2.  <br>
3.  <br>
4.  <br>
5.  <br>

 + 동시에 Eye Tracking 진행 <br>