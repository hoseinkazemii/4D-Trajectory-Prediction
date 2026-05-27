using UnityEngine;
using System.IO;

public class HookHeightLinearEncoderSimulator : MonoBehaviour
{
    private float resolution = 5f; // 5 micrometers per pulse
    private float previousHookPosition = 0f; // Previous hook position
    private float currentHookPosition = 0f; // Current hook position
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

    public CSVWriter csvWriter; // Assign this in the Inspector

    private float hookHeight = 0f; // Variable to store the calculated hook height

    void Start()
    {
        // Initialize previous position with the starting position
        previousHookPosition = craneScript.hook;
        distancePerPulse = resolution / 10f; // Converting micrometers to millimeters (multiplied by 100)
        accumulatedDistanceB = distancePerPulse / 2f;
    }

    void Update()
    {
        // Access the hook variable from the crane_animate2 script
        currentHookPosition = craneScript.hook;

        // Calculate the change in position since the last frame
        deltaDistance = currentHookPosition - previousHookPosition;

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

        // Update hook height based on signal changes using quadrature decoding
        if (signalAChanged)
        {
            direction = CalculateDirection(signalA, signalB, prevSignalA);
            hookHeight += direction * distancePerPulse;

            signalAChanged = false;

            // Update the centralized CSV writer with the latest rotary encoder signals
            csvWriter.UpdateHookHeightEncoderSignals(signalA, signalB, hookHeight);

            // Raise the event
            EncoderSignalChangeNotifier.SignalAChanged();
        }

        // Update the previous signals
        prevSignalA = signalA;
        // Update the previous position
        previousHookPosition = currentHookPosition;
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


/// Hook Height linear encoder data synthesizer calculator before combining the csv outputs into one export

// using UnityEngine;
// using System.IO;

// public class HookHeightLinearEncoderSimulator : MonoBehaviour
// {
//     private float resolution = 5f; // 5 micrometers per pulse
//     private float previousHookPosition = 0f; // Previous hook position
//     private float currentHookPosition = 0f; // Current hook position
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

//     private float hookHeight = 0f; // Variable to store the calculated hook height

//     void Start()
//     {
//         // Generate timestamp for the file name
//         string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");

//         // Set the file path to the Desktop with timestamp
//         filePath = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"HookHeightLinearEncoderData_{timestamp}.csv");

//         // Initialize previous position with the starting position
//         previousHookPosition = craneScript.hook;
//         distancePerPulse = resolution / 10f; // Converting micrometers to millimeters (multiplied by 100)
//         accumulatedDistanceB = distancePerPulse / 2f;

//         // Write the header to the CSV file
//         if (!File.Exists(filePath))
//         {
//             File.WriteAllText(filePath, "TimeFrame,SignalAHookHeightLinearEncoder,SignalBHookHeightLinearEncoder,HookHeight\n");
//         }

//         // Record the start time
//         startTime = Time.time;
//     }

//     void Update()
//     {
//         // Access the hook variable from the crane_animate2 script
//         currentHookPosition = craneScript.hook;

//         // Calculate the change in position since the last frame
//         deltaDistance = currentHookPosition - previousHookPosition;

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

//         // Update hook height based on signal changes using quadrature decoding
//         if (signalAChanged)
//         {
//             direction = CalculateDirection(signalA, signalB, prevSignalA);
//             hookHeight += direction * distancePerPulse;

//             signalAChanged = false;

//             // Append the current timeframe, Signal A, Signal B, and Hook Height values to the CSV file
//             float currentTimeFrame = Time.time - startTime;
//             string newLine = $"{currentTimeFrame:F2},{(signalA ? 1 : 0)},{(signalB ? 1 : 0)},{hookHeight}\n";
//             File.AppendAllText(filePath, newLine);
//         }

//         // Update the previous signals
//         prevSignalA = signalA;
//         // Update the previous position
//         previousHookPosition = currentHookPosition;
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