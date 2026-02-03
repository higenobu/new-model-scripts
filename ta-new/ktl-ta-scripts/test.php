<?php
// Define the FastAPI endpoint
$url = 'http://localhost:8000/analyze';

// Data to send in the POST request
$data = [
    "texts" => ["This is fantastic.", "I feel scared."]
];

// Initialize a cURL session
$ch = curl_init($url);

// Set request options
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);  // Return response as a string
curl_setopt($ch, CURLOPT_POST, true);            // Use POST method
curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']); // JSON headers
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data)); // JSON payload

// Execute the cURL request and get the response
$response = curl_exec($ch);

// Check for errors
if(curl_errno($ch)) {
    echo 'Error: ' . curl_error($ch);
} else {
    // Parse and display the JSON response
    $json = json_decode($response, true);
    echo "API Response:\n";
    print_r($json);
}

// Close the cURL session
curl_close($ch);
