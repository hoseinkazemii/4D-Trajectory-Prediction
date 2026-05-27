using UnityEngine;
using System.IO;

public class RadiusLinearEncoderSimulator : MonoBehaviour
{
    private float resolution = 5f; // 5 micrometers per pulse
    private float previousDollyPosition = 10f; // Previous trolley position
    private float currentDollyPosition = 10f; // Current trolley position
    private float deltaDistance;
    private float accumulatedDistanceA;
    private float accumulatedDistanceB;
    private float distancePerPulse;

    public crane_animate2 craneScript; // Reference to the crane_animate2 script (set in the Inspector)

    private bool signalA = false;
    private bool signalB = false;
    private bool prevSignalA = false;
    private bool signalAChanged = false;
    private int direction = 0;

    private string filePath;
    private float startTime;

    public CSVWriter csvWriter; // Assign this in the Inspector

    private float trolleyDistance = 10f; // Variable to store the calculated trolley distance

    void Start()
    {
        // Initialize previous position with the starting position
        previousDollyPosition = craneScript.dolly;
        distancePerPulse = resolution / 10f; // Converting micrometers to millimeters (multiplied by 100)
        accumulatedDistanceB = distancePerPulse / 2f;
    }

    void Update()
    {
        // Access the dolly variable from the crane_animate2 script
        currentDollyPosition = craneScript.dolly;

        // Calculate the change in position since the last frame
        deltaDistance = currentDollyPosition - previousDollyPosition;

        // Accumulate the deltaDistance
        accumulatedDistanceA += deltaDistance;
        accumulatedDistanceB += deltaDistance;

        // Check if the accumulated distance exceeds the distance per pulse
        // Handling Signal A
        if (Mathf.Abs(accumulatedDistanceA) >= distancePerPulse)
        {
            accumulatedDistanceA %= distancePerPulse;
            // Toggle the state of Signal A
            signalA = !signalA;
            signalAChanged = true;
        }

        // Handling Signal B
        if (Mathf.Abs(accumulatedDistanceB) >= distancePerPulse)
        {
            accumulatedDistanceB %= distancePerPulse;
            // Toggle the state of Signal B
            signalB = !signalB;
        }

        // Update trolley distance based on signal changes using quadrature decoding
        if (signalAChanged)
        {
            direction = CalculateDirection(signalA, signalB, prevSignalA);
            trolleyDistance += direction * distancePerPulse;

            signalAChanged = false;

            // Update the centralized CSV writer with the latest rotary encoder signals
            csvWriter.UpdateRadiusEncoderSignals(signalA, signalB, trolleyDistance);

            // Raise the event
            EncoderSignalChangeNotifier.SignalAChanged();
        }

        // Update the previous signals
        prevSignalA = signalA;
        // Update the previous position
        previousDollyPosition = currentDollyPosition;
    }

    int CalculateDirection(bool signalA, bool signalB, bool prevSignalA)
    {
        // Simplified logic to determine the direction
        if (signalA != prevSignalA) // Signal A has changed
        {
            return (signalA == signalB) ? 1 : -1;
        }
        return 0;
    }
}



/// Trolley Movement (Radius) linear encoder data synthesizer calculator before combining the csv outputs into one export

// using UnityEngine;
// using System.IO;

// public class RadiusLinearEncoderSimulator : MonoBehaviour
// {
//     private float resolution = 5f; // 5 micrometers per pulse
//     private float previousDollyPosition = 10f; // Previous trolley position
//     private float currentDollyPosition = 10f; // Current trolley position
//     private float deltaDistance;
//     private float accumulatedDistanceA;
//     private float accumulatedDistanceB;
//     private float distancePerPulse;

//     public crane_animate2 craneScript; // Reference to the crane_animate2 script (set in the Inspector)

//     private bool signalA = false;
//     private bool signalB = false;
//     private bool prevSignalA = false;
//     private bool signalAChanged = false;
//     private int direction = 0;

//     private string filePath;
//     private float startTime;

//     private float trolleyDistance = 10f; // Variable to store the calculated trolley distance

//     void Start()
//     {
//         // Generate timestamp for the file name
//         string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");

//         // Set the file path to the Desktop with timestamp
//         filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"RadiusLinearEncoderData_{timestamp}.csv");

//         // Initialize previous position with the starting position
//         previousDollyPosition = craneScript.dolly;
//         distancePerPulse = resolution / 10f; // Converting micrometers to millimeters (multiplied by 100)
//         accumulatedDistanceB = distancePerPulse / 2f;

//         // Write the header to the CSV file
//         if (!File.Exists(filePath))
//         {
//             File.WriteAllText(filePath, "TimeFrame,SignalARadiusLinearEncoder,SignalBRadiusLinearEncoder,Radius\n");
//         }

//         // Record the start time
//         startTime = Time.time;
//     }

//     void Update()
//     {
//         // Access the dolly variable from the crane_animate2 script
//         currentDollyPosition = craneScript.dolly;

//         // Calculate the change in position since the last frame
//         deltaDistance = currentDollyPosition - previousDollyPosition;

//         // Accumulate the deltaDistance
//         accumulatedDistanceA += deltaDistance;
//         accumulatedDistanceB += deltaDistance;

//         // Check if the accumulated distance exceeds the distance per pulse
//         // Handling Signal A
//         if (Mathf.Abs(accumulatedDistanceA) >= distancePerPulse)
//         {
//             accumulatedDistanceA %= distancePerPulse;
//             // Toggle the state of Signal A
//             signalA = !signalA;
//             signalAChanged = true;
//         }

//         // Handling Signal B
//         if (Mathf.Abs(accumulatedDistanceB) >= distancePerPulse)
//         {
//             accumulatedDistanceB %= distancePerPulse;
//             // Toggle the state of Signal B
//             signalB = !signalB;
//         }

//         // Update trolley distance based on signal changes using quadrature decoding
//         if (signalAChanged)
//         {
//             direction = CalculateDirection(signalA, signalB, prevSignalA);
//             trolleyDistance += direction * distancePerPulse;

//             signalAChanged = false;

//             // Append the current timeframe, Signal A, Signal B, and Trolley Distance values to the CSV file
//             float currentTimeFrame = Time.time - startTime;
//             string newLine = $"{currentTimeFrame:F2},{(signalA ? 1 : 0)},{(signalB ? 1 : 0)},{trolleyDistance}\n";
//             File.AppendAllText(filePath, newLine);
//         }

//         // Update the previous signals
//         prevSignalA = signalA;
//         // Update the previous position
//         previousDollyPosition = currentDollyPosition;
//     }

//     int CalculateDirection(bool signalA, bool signalB, bool prevSignalA)
//     {
//         // Simplified logic to determine the direction
//         if (signalA != prevSignalA) // Signal A has changed
//         {
//             return (signalA == signalB) ? 1 : -1;
//         }
//         return 0;
//     }
// }

