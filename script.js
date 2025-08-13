const foodList = ['拉面', '盖饭', '麻辣烫', '沙拉', '汉堡', '饺子', 'KFC', '麦当劳', '必胜客'];
const colors = ["#FBE4D8", "#DFCCF1", "#B9E2F5", "#F5DDE0", "#F7E4A9", "#D4F0F0", "#F5D4C1", "#C8E6C9", "#FFF9C4"];

const canvas = document.getElementById('wheelCanvas');
const ctx = canvas.getContext('2d');
const choiceBtn = document.getElementById('choiceBtn');

let currentRotation = 0;
let isSpinning = false;

function drawWheel() {
    const numOptions = foodList.length;
    const arcSize = (2 * Math.PI) / numOptions;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = canvas.width / 2 - 10;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(currentRotation);
    ctx.translate(-centerX, -centerY);

    for (let i = 0; i < numOptions; i++) {
        const angle = i * arcSize;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, angle, angle + arcSize, false);
        ctx.lineTo(centerX, centerY);
        ctx.fillStyle = colors[i % colors.length];
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(angle + arcSize / 2);
        ctx.textAlign = 'right';
        ctx.fillStyle = '#555';
        ctx.font = 'bold 18px "Ma Shan Zheng"';
        ctx.fillText(foodList[i], radius - 15, 10);
        ctx.restore();
    }
    ctx.restore();

    // Draw center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 20, 0, 2 * Math.PI, false);
    ctx.fillStyle = '#ff8177';
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 4;
    ctx.stroke();
}

function spin() {
    if (isSpinning) return;
    isSpinning = true;
    choiceBtn.disabled = true;

    const randomSpins = Math.floor(Math.random() * 5) + 5; // 5 to 10 full spins
    const targetAngle = Math.random() * 2 * Math.PI; // Random final angle
    const totalRotation = randomSpins * 2 * Math.PI + targetAngle;

    canvas.style.transition = 'transform 5s cubic-bezier(0.25, 0.1, 0.25, 1)';
    canvas.style.transform = `rotate(${totalRotation}rad)`;

    setTimeout(() => {
        const finalRotation = totalRotation % (2 * Math.PI);
        const arcSize = (2 * Math.PI) / foodList.length;
        
        // Correct the angle for the pointer at the top (-90 degrees)
        const correctedAngle = finalRotation + (Math.PI / 2);
        const winningSegment = Math.floor(((2 * Math.PI) - (correctedAngle % (2 * Math.PI))) / arcSize) % foodList.length;

        setTimeout(() => {
            alert(`今天中午吃：${foodList[winningSegment]}!`);
            canvas.style.transition = 'none';
            canvas.style.transform = `rotate(${finalRotation}rad)`;
            isSpinning = false;
            choiceBtn.disabled = false;
        }, 500);

    }, 5000);
}

choiceBtn.addEventListener('click', spin);

drawWheel();