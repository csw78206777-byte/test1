document.addEventListener('DOMContentLoaded', () => {
    const choiceBtn = document.getElementById('choiceBtn');
    const resultDiv = document.getElementById('result');

    const foodList = [
        '拉面',
        '盖饭',
        '麻辣烫',
        '沙拉',
        '汉堡',
        '饺子',
        '披萨',
        '寿司'
    ];

    choiceBtn.addEventListener('click', () => {
        const randomIndex = Math.floor(Math.random() * foodList.length);
        const choice = foodList[randomIndex];
        resultDiv.textContent = `就吃${choice}！`;
        // 重新触发动画
        resultDiv.style.animation = 'none';
        resultDiv.offsetHeight; /* 触发重排 */
        resultDiv.style.animation = null;
    });
});