// ========== ECharts 全局主题注册（紫色系，与UI统一） ==========
if (typeof echarts !== 'undefined') {
    echarts.registerTheme('yogurt', {
        color: ['#6366f1', '#8b5cf6', '#a78bfa', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#c084fc', '#2dd4bf'],
        backgroundColor: 'transparent',
        textStyle: { fontFamily: 'Inter, sans-serif', color: '#334155' },
        title: { textStyle: { fontWeight: 'normal', color: '#1e293b' } },
        tooltip: { backgroundColor: 'rgba(255,255,255,0.96)', borderColor: '#e2e8f0' }
    });
}

// 设置 AJAX 全局 CSRF 请求头
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
            if (typeof csrf_token !== 'undefined') {
                xhr.setRequestHeader("X-CSRFToken", csrf_token);
            }
        }
    }
});

$(document).ready(function () {
    setTimeout(function () {
        $('.alert:not(#filter-badge)').fadeOut('slow');
    }, 4000);
    $('[onclick^="return confirm"]').on('click', function (e) {
        if (!confirm('确定要执行此操作吗？')) {
            e.preventDefault();
            return false;
        }
    });
});

function showToast(message, type = 'success') {
    var bgClass = type === 'success' ? 'bg-success' : (type === 'danger' ? 'bg-danger' : 'bg-info');
    var toastHtml = `<div class="toast align-items-center text-white ${bgClass} border-0" role="alert" aria-live="assertive" aria-atomic="true" data-bs-autohide="true" data-bs-delay="3000">
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    </div>`;
    $('.toast-container').append(toastHtml);
    var toast = new bootstrap.Toast($('.toast').last()[0]);
    toast.show();
    $('.toast').last().on('hidden.bs.toast', function() { $(this).remove(); });
}

function formatNumber(num, decimal = 2) {
    if (num === null || num === undefined || isNaN(num)) return '-';
    return parseFloat(num).toFixed(decimal);
}

function setLoading(selector, text = '处理中...') {
    $(selector).prop('disabled', true).html('<i class="fa fa-spinner fa-spin me-2"></i>' + text);
}

function removeLoading(selector, originText) {
    $(selector).prop('disabled', false).html(originText);
}

function drawPieChart(domId, data, title = '') {
    var chart = echarts.init(document.getElementById(domId), 'yogurt');
    chart.setOption({
        title: { text: title, left: 'center', textStyle: { fontSize: 14 } },
        tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
        series: [{
            type: 'pie',
            radius: ['40%', '70%'],
            data: data,
            label: { show: true, formatter: '{b}: {d}%' },
            itemStyle: { borderRadius: 6 }
        }]
    });
    window.addEventListener('resize', function () { chart.resize(); });
    return chart;
}

function drawBarChart(domId, xData, yData, yName = '') {
    var chart = echarts.init(document.getElementById(domId), 'yogurt');
    chart.setOption({
        tooltip: { trigger: 'axis' },
        grid: { containLabel: true },
        xAxis: { type: 'category', data: xData, axisLabel: { rotate: 20 } },
        yAxis: { type: 'value', name: yName },
        series: [{
            type: 'bar',
            data: yData,
            itemStyle: { borderRadius: [4, 4, 0, 0] }
        }]
    });
    window.addEventListener('resize', function () { chart.resize(); });
    return chart;
}

function drawLineChart(domId, xData, yData, yName = '') {
    var chart = echarts.init(document.getElementById(domId), 'yogurt');
    chart.setOption({
        tooltip: { trigger: 'axis' },
        grid: { containLabel: true },
        xAxis: { type: 'category', data: xData },
        yAxis: { type: 'value', name: yName },
        series: [{
            type: 'line',
            data: yData,
            smooth: true,
            lineStyle: { width: 2 }
        }]
    });
    window.addEventListener('resize', function () { chart.resize(); });
    return chart;
}

function initDataTable(selector, options) {
    var defaults = {
        pageLength: 25,
        lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "全部"]],
        language: {
            "sProcessing": "处理中...",
            "sLengthMenu": "显示 _MENU_ 项",
            "sZeroRecords": "没有匹配结果",
            "sInfo": "显示第 _START_ 至 _END_ 项，共 _TOTAL_ 项",
            "sInfoEmpty": "显示第 0 至 0 项，共 0 项",
            "sInfoFiltered": "(由 _MAX_ 项过滤)",
            "sSearch": "全局搜索：",
            "oPaginate": {
                "sFirst": "首页",
                "sPrevious": "上页",
                "sNext": "下页",
                "sLast": "末页"
            }
        },
        columnDefs: []
    };
    return $(selector).DataTable($.extend(true, {}, defaults, options));
}