"""
Shared CSS styling for validation HTML files to ensure consistent theming
across all validation reports in the FRBAtl TariffPricePulse project.
"""

def get_shared_css():
    """Returns consistent CSS styling for all validation HTML files"""
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            line-height: 1.6;
            color: #2c3e50;
        }
        
        .header {
            text-align: center;
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 30px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.2em;
            font-weight: 300;
        }
        
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .back-button {
            display: inline-block;
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            text-decoration: none;
            padding: 12px 20px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 14px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(231, 76, 60, 0.3);
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
        }
        
        .back-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
            color: white;
            text-decoration: none;
        }
        
        .back-button::before {
            content: "← ";
            margin-right: 5px;
        }
        
        .tabs {
            display: flex;
            background: white;
            border-radius: 10px 10px 0 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .tabs button {
            background-color: #ecf0f1;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 18px 24px;
            transition: all 0.3s ease;
            color: #2c3e50;
            font-size: 16px;
            font-weight: 500;
            flex: 1;
            border-right: 1px solid #bdc3c7;
        }
        
        .tabs button:last-child {
            border-right: none;
        }
        
        .tabs button:hover {
            background-color: #d5dbdb;
        }
        
        .tabs button.active {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
        }
        
        .tabcontent {
            display: none;
            padding: 30px;
            background: white;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .tabcontent.active {
            display: block;
        }
        
        .tabcontent h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
            margin-bottom: 20px;
        }
        
        .summary, .intro-text {
            background: linear-gradient(135deg, #ebf3fd, #e8f4f8);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            border-left: 4px solid #3498db;
        }
        
        .summary p, .intro-text p {
            margin: 0 0 10px 0;
            color: #34495e;
        }
        
        .summary p:last-child, .intro-text p:last-child {
            margin-bottom: 0;
        }
        
        .plot-container {
            margin-bottom: 30px;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .plot-container h3 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        
        .scale-toggle {
            margin-bottom: 15px;
            padding: 12px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 6px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        
        .scale-toggle label {
            margin-right: 20px;
            color: #495057;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
        }
        
        .scale-toggle input[type="radio"] {
            margin-right: 8px;
            accent-color: #3498db;
        }
        
        .plot-iframe {
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .plot-iframe.hidden {
            display: none;
        }
        
        iframe {
            width: 100% !important;
            height: 600px;
            border: none;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            min-width: 100%;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
            color: #2c3e50;
        }
        
        th {
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        
        tr:hover {
            background-color: #f8f9fa;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
        }
        
        /* Plot styling adjustments */
        div[id$="-plot"] {
            margin: 15px 0;
            min-height: 500px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            .header {
                padding: 20px 15px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
            
            .tabs {
                flex-direction: column;
            }
            
            .tabs button {
                border-right: none;
                border-bottom: 1px solid #bdc3c7;
            }
            
            .tabs button:last-child {
                border-bottom: none;
            }
            
            .tabcontent {
                padding: 20px;
            }
            
            .plot-iframe, iframe {
                height: 400px;
            }
        }
    """

def get_shared_javascript():
    """Returns consistent JavaScript for tab functionality"""
    return """
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            
            // Hide all tab content
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
                tabcontent[i].classList.remove("active");
            }
            
            // Remove active class from all tab buttons
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].classList.remove("active");
            }
            
            // Show the selected tab and mark button as active
            document.getElementById(tabName).style.display = "block";
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }
        
        function toggleScale(region, scale) {
            var regularFrame = document.getElementById(region + '_regular');
            var logFrame = document.getElementById(region + '_log');
            
            if (scale === 'regular') {
                regularFrame.classList.remove('hidden');
                logFrame.classList.add('hidden');
            } else {
                regularFrame.classList.add('hidden');
                logFrame.classList.remove('hidden');
            }
        }
        
        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            // Click the first tab by default
            var defaultTab = document.getElementById("defaultOpen");
            if (defaultTab) {
                defaultTab.click();
            }
        });
    """