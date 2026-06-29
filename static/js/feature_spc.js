// static/js/feature_spc.js
let iChart, mrChart;
let currentData = null;

$(function() {
    // 品项变化时动态加载CTQ列表（根据品项过滤）
    $('#productItemSelect').change(function() {
        const productItem = $(this).val();
        loadCtqList(productItem);
        $('#featureSelect').html('<option value="">-- 请先选择CTQ --</option>').prop('disabled', true);
    });

    // CTQ变化时加载特征列表
    $('#ctqSelect').change(function() {
        const ctqId = $(this).val();
        const productItem = $('#productItemSelect').val();
        if (!ctqId) return;
        loadFeatures(ctqId, productItem);
    });

    // 加载按钮
    $('#loadBtn').click(function() {
        const ctqId = $('#ctqSelect').val();
        const featureName = $('#featureSelect').val();
        const productItem = $('#productItemSelect').val();
        const startDate = $('#startDate').val();
        const endDate = $('#endDate').val();
        const rules = $('.rule-check:checked').map(function() { return this.value; }).get().join(',');

        if (!ctqId || !featureName) {
            alert('请完整选择 CTQ 和影响因素');
            return;
        }

        const params = { ctq_id: ctqId, feature_name: featureName, product_item: productItem, start_date: startDate, end_date: endDate, rules: rules };
        $('#loadBtn').prop('disabled', true).html('<i class="fa fa-spinner fa-spin"></i> 加载中...');
        $.getJSON('/feature-monitor/api/data', params)
            .done(function(res) {
                if (res.error) { alert(res.error); return; }
                currentData = res;
                drawCharts(res);
                updateAlarmTable(res);
                fetchAdvice(res);
                $('#chartInfo').text(`${res.feature_name} | 均值=${res.mean} | 标准差=${res.std}`);
            })
            .fail(function() { alert('数据加载失败'); })
            .always(function() { $('#loadBtn').prop('disabled', false).html('<i class="fa fa-play"></i> 绘制控制图'); });
    });

    $('#resetRules').click(function() { $('.rule-check').prop('checked', true); });
});

function loadCtqList(productItem) {
    let allCtqs = window.ctqList || [];
    let filtered = allCtqs;
    if (productItem) {
        filtered = allCtqs.filter(c => !c.product_item || c.product_item === productItem);
    }
    let html = '<option value="">-- 请选择CTQ --</option>';
    filtered.forEach(c => {
        html += `<option value="${c.ctq_id}">${c.ctq_name} (${c.product_item || '通用'})</option>`;
    });
    $('#ctqSelect').html(html);
}

function loadFeatures(ctqId, productItem) {
    $.getJSON('/feature-monitor/api/features', { ctq_id: ctqId, product_item: productItem })
        .done(function(res) {
            let html = '<option value="">-- 请选择特征 --</option>';
            (res.features || []).forEach(f => { html += `<option value="${f}">${f}</option>`; });
            $('#featureSelect').html(html).prop('disabled', false);
        })
        .fail(function() { alert('特征列表加载失败'); });
}

function drawCharts(data) {
    const labels = data.labels || data.xbar.map((_, i) => i+1);
    // I 图
    if (iChart) iChart.dispose();
    iChart = echarts.init(document.getElementById('iChart'));
    iChart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['实测值', 'UCL', 'CL', 'LCL'], left: 'left' },
        grid: { containLabel: true, left: '5%', right: '5%' },
        xAxis: { type: 'category', data: labels, axisLabel: { rotate: 30, interval: 0 } },
        yAxis: { type: 'value', name: '特征值' },
        series: [
            { name: '实测值', type: 'line', data: data.xbar, symbol: 'circle', symbolSize: 6, lineStyle: { color: '#6366f1' }, markPoint: { data: (data.alarm_x||[]).map(i => ({ coord: [i, data.xbar[i]], name: '异常', itemStyle: { color: '#ef4444' } })) } },
            { name: 'UCL', type: 'line', data: Array(labels.length).fill(data.ucl_x), lineStyle: { type: 'dashed', color: '#ef4444' }, symbol: 'none' },
            { name: 'CL', type: 'line', data: Array(labels.length).fill(data.cl_x), lineStyle: { type: 'dashed', color: '#10b981' }, symbol: 'none' },
            { name: 'LCL', type: 'line', data: Array(labels.length).fill(data.lcl_x), lineStyle: { type: 'dashed', color: '#ef4444' }, symbol: 'none' }
        ]
    });

    // MR 图
    if (mrChart) mrChart.dispose();
    mrChart = echarts.init(document.getElementById('mrChart'));
    const mrLabels = labels.slice(1);
    mrChart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['移动极差', 'UCL', 'CL'] },
        xAxis: { type: 'category', data: mrLabels },
        yAxis: { type: 'value', name: '移动极差' },
        series: [
            { name: '移动极差', type: 'line', data: data.mr, symbol: 'circle', symbolSize: 6, lineStyle: { color: '#f59e0b' } },
            { name: 'UCL', type: 'line', data: Array(mrLabels.length).fill(data.ucl_r), lineStyle: { type: 'dashed', color: '#ef4444' }, symbol: 'none' },
            { name: 'CL', type: 'line', data: Array(mrLabels.length).fill(data.cl_r), lineStyle: { type: 'dashed', color: '#10b981' }, symbol: 'none' }
        ]
    });
}

function updateAlarmTable(data) {
    const container = $('#alarmTable');
    if (!data.rules_violations || data.rules_violations.length === 0) {
        container.html('<div class="text-muted small">未检测到异常点，过程受控。</div>');
        return;
    }
    let html = '<table class="table table-sm table-striped"><thead><tr><th>异常描述</th><tr></thead><tbody>';
    data.rules_violations.forEach(v => { html += `<tr><td>${escapeHtml(v)}</td>`; });
    html += '</tbody></table>';
    container.html(html);
}

function fetchAdvice(data) {
    $.ajax({
        url: '/feature-monitor/api/advice',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(data),
        success: function(res) {
            $('#advicePanel').html(`<div class="small">${res.advice}</div>`);
        },
        error: function() { $('#advicePanel').html('<p class="text-danger small">建议生成失败</p>'); }
    });
}

function escapeHtml(str) {
    return String(str).replace(/[&<>]/g, function(m) {
        if (m === '&') return '&amp;';
        if (m === '<') return '&lt;';
        if (m === '>') return '&gt;';
        return m;
    });
}