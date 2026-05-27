using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Metrics_Calculator_PowerLine : MonoBehaviour
{
    public Transform grabbableObject; // Assign the grabbable object in the Inspector
    public Transform unloadingArea; // Assign the unloading area in the Inspector
    public Transform loadSwayMeasurementReferencePoint; // Assign the reference point in the Inspector
    private Transform craneHook;
    private Vector3 previousPosition;
    private bool loadingStarted = false;
    private bool firstFrameAfterLoadingStarted = true;
    private float totalPathLength = 0f;
    private List<PathPoint> pathCoordinates = new List<PathPoint>(); // Record positions and times for the path
    private string dataExportPath = "C:/Users/mzk6120/Desktop";
    private string metricsFilePath;
    private float preciseness;
    private List<float> loadSwayMeasurements = new List<float>(); // Record sway measurements
    public MoveObject moveObject;
    private float taskStartTime;
    private float taskEndTime;
    private float taskCompletionTime;
    public CSVWriter csvWriter; // Assign this in the Inspector
    // Store collision data
    private List<string> collisionLogs = new List<string>();
    private bool taskEnded = false;

    void Start()
    {
        craneHook = transform;
        // Subscribe to the event
        EncoderSignalChangeNotifier.OnSignalAChanged += WriteMetrics;
        // Subscribe to collision event
        DustCloudTriggerPowerLine.OnCollisionLogged += LogCollision;

        // Start invoking the PathLengthCalculation method every second
        InvokeRepeating("PathLengthCalculation", 0f, 1f);
    }

    public void InitializeMetricsCalculator(string scenarioName)
    {
        string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");
        metricsFilePath = Path.Combine(dataExportPath, $"Metrics_{scenarioName}_{timestamp}.txt");
    }

    void OnDestroy()
    {
        // Unsubscribe from the event
        EncoderSignalChangeNotifier.OnSignalAChanged -= WriteMetrics;
        DustCloudTriggerPowerLine.OnCollisionLogged -= LogCollision;

        // Cancel any scheduled invokes
        CancelInvoke("PathLengthCalculation");
    }

    private void WriteMetrics()
    {
        float sway = 0f;
        if (grabbableObject != null && grabbableObject.parent == craneHook)
        {
            sway = Vector3.Distance(grabbableObject.position, loadSwayMeasurementReferencePoint.position);
            loadSwayMeasurements.Add(sway);
        }
        // Update the centralized CSV writer with the latest metrics
        csvWriter.UpdateMetrics(Time.time, grabbableObject.position.x, grabbableObject.position.y, grabbableObject.position.z, sway, loadingStarted);
    }

    void FixedUpdate()
    {
        taskStartTime = crane_animate2.taskStartTime;
        taskEndTime = DustCloudTriggerPowerLine.taskEndTime;
        taskEnded = DustCloudTriggerPowerLine.taskEnded;  // Use taskEnded from DustCloudTrigger
        loadingStarted = !moveObject.canGrab;

        // Stop recording path length once the task has ended
        if (taskEnded)
        {
            CancelInvoke("PathLengthCalculation");
            return;
        }

        // Start the path calculation on loading
        if (loadingStarted && firstFrameAfterLoadingStarted)
        {
            previousPosition = craneHook.position;
            firstFrameAfterLoadingStarted = false;
        }
    }

    // Method called at regular intervals to calculate path length
    private void PathLengthCalculation()
    {
        if (loadingStarted)
        {
            float distance = Vector3.Distance(previousPosition, craneHook.position);
            totalPathLength += distance;
            
            // Store the position and time as a new PathPoint
            pathCoordinates.Add(new PathPoint(craneHook.position, Time.time));
            previousPosition = craneHook.position;
        }
    }

    // Handle the collision logging
    private void LogCollision(string objectName, float collisionTime, Vector3 collisionPosition)
    {
        string log = $"Collision with: {objectName}, Time: {collisionTime:F2}, Position: {collisionPosition}";
        collisionLogs.Add(log);
    }

    public void ExportMetrics()
    {
        taskCompletionTime = taskEndTime - taskStartTime;
        using (StreamWriter writer = new StreamWriter(metricsFilePath))
        {
            writer.WriteLine("Path Length: " + (totalPathLength / 2).ToString("F2") + " meters"); // totalPathLength is divided by 2 because of scaling differences between Unity and real-world
            writer.WriteLine(new string('-', 40));
            // Write path coordinates to the file
            writer.WriteLine("Path Coordinates:");
            foreach (PathPoint point in pathCoordinates)
            {
                writer.WriteLine($"{point.Position} | Time: {point.Time:F2}"); // Format time to 2 decimal places
            }
            writer.WriteLine(new string('-', 40));
            // Write task completion time to the file
            writer.WriteLine("Task Completion Time: " + taskCompletionTime.ToString("F2") + " seconds");
            writer.WriteLine("Start Time: " + "Second " + taskStartTime.ToString("F2"));
            writer.WriteLine("End Time: " + "Second " + taskEndTime.ToString("F2"));
            writer.WriteLine(new string('-', 40));
            // Write Preciseness
            if (grabbableObject != null && unloadingArea != null)
            {
                preciseness = Vector3.Distance(grabbableObject.position, unloadingArea.position) / 5; // Divided by 5 because of scaling differences
            }
            else
            {
                Debug.LogError("GrabbableObject or UnloadingArea is not assigned!");
            }
            writer.WriteLine("Preciseness: " + preciseness.ToString("F2") + " meters");
            writer.WriteLine(new string('-', 40));
            // Write collision logs
            writer.WriteLine("Collision Logs:");
            foreach (string log in collisionLogs)
            {
                writer.WriteLine(log);
            }
        }
    }
}







// using System.Collections.Generic;
// using System.IO;
// using UnityEngine;

// public class Metrics_Calculator : MonoBehaviour
// {
//     public Transform grabbableObject; // Assign the grabbable object in the Inspector
//     public Transform unloadingArea; // Assign the unloading area in the Inspector
//     public Transform loadSwayMeasurementReferencePoint; // Assign the reference point in the Inspector

//     private Transform craneHook;
//     private Vector3 previousPosition;
//     private bool loadingStarted = false;
//     private bool firstFrameAfterLoadingStarted = true;
//     private float totalPathLength = 0f;
//     private List<Vector3> pathCoordinates = new List<Vector3>(); // Record positions for the path
//     private float elapsedTime = 0f;
//     private float measurementInterval = 1f; // Time interval in seconds
//     private string dataExportPath = "C:/Users/mzk6120/Desktop";
//     private string metricsFilePath;

//     private float preciseness;

//     private List<float> loadSwayMeasurements = new List<float>(); // Record sway measurements

//     public MoveObject moveObject;

//     private float taskStartTime;
//     private float taskEndTime;
//     private float taskCompletionTime;

//     public CSVWriter csvWriter; // Assign this in the Inspector

//     // Store collision data
//     private List<string> collisionLogs = new List<string>();

//     void Start()
//     {
//         craneHook = transform;

//         // Subscribe to the event
//         EncoderSignalChangeNotifier.OnSignalAChanged += WriteMetrics;

//         // Subscribe to collision event
//         DustCloudTrigger.OnCollisionLogged += LogCollision;
//     }

//     public void InitializeMetricsCalculator(string scenarioName)
//     {
//         string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");
//         metricsFilePath = Path.Combine(dataExportPath, $"Metrics_{scenarioName}_{timestamp}.txt");
//     }

//     void OnDestroy()
//     {
//         // Unsubscribe from the event
//         EncoderSignalChangeNotifier.OnSignalAChanged -= WriteMetrics;
//         DustCloudTrigger.OnCollisionLogged -= LogCollision;
//     }

//     private void WriteMetrics()
//     {
//         float sway = 0f;
//         if (grabbableObject != null && grabbableObject.parent == craneHook)
//         {
//             sway = Vector3.Distance(grabbableObject.position, loadSwayMeasurementReferencePoint.position);
//             loadSwayMeasurements.Add(sway);
//         }

//         // Update the centralized CSV writer with the latest metrics
//         csvWriter.UpdateMetrics(Time.time, grabbableObject.position.x, grabbableObject.position.y, grabbableObject.position.z, sway, loadingStarted);
//     }

//     void FixedUpdate()
//     {
//         taskStartTime = crane_animate2.taskStartTime;
//         taskEndTime = DustCloudTrigger.taskEndTime;

//         loadingStarted = !moveObject.canGrab;
//         // Calculate path length during loading and save the result
//         if (loadingStarted)
//         {
//             if (firstFrameAfterLoadingStarted)
//             {
//                 previousPosition = craneHook.position;
//                 firstFrameAfterLoadingStarted = false;
//             }

//             elapsedTime += Time.deltaTime;

//             // Measure path length and record load position and sway at the specified interval
//             if (elapsedTime >= measurementInterval)
//             {
//                 Debug.Log("Adding a position..."); // Fix this bug: After loading is finished, it should no more log the position of the object. You can use taskEndTime to stop logging in the FixedUpdate()
//                 float distance = Vector3.Distance(previousPosition, craneHook.position);
//                 totalPathLength += distance;
//                 pathCoordinates.Add(craneHook.position); // Record crane_hook position for the path
//                 previousPosition = craneHook.position;

//                 elapsedTime = 0f; // Reset elapsed time after recording data
//             }
//         }
//     }

//     // Handle the collision logging
//     private void LogCollision(string objectName, float collisionTime, Vector3 collisionPosition)
//     {
//         string log = $"Collision with: {objectName}, Time: {collisionTime:F2}, Position: {collisionPosition}";
//         collisionLogs.Add(log);
//     }

//     public void ExportMetrics()
//     {
//         taskCompletionTime = taskEndTime - taskStartTime;

//         using (StreamWriter writer = new StreamWriter(metricsFilePath))
//         {
//             writer.WriteLine("Path Length: " + (totalPathLength / 2).ToString("F2") + " meters"); // totalPathLength is divided by 2 because of scaling differences between Unity and real-world

//             // Add a separator line
//             writer.WriteLine(new string('-', 40));

//             // Write path coordinates to the file
//             writer.WriteLine("Path Coordinates:");
//             foreach (Vector3 coordinate in pathCoordinates)
//             {
//                 writer.WriteLine(coordinate.ToString());
//             }

//             writer.WriteLine(new string('-', 40));

//             // Write task completion time to the file
//             writer.WriteLine("Task Completion Time: " + taskCompletionTime.ToString("F2") + " seconds");
//             writer.WriteLine("Start Time: " + "Second " + taskStartTime.ToString("F2"));
//             writer.WriteLine("End Time: " + "Second " + taskEndTime.ToString("F2"));

//             writer.WriteLine(new string('-', 40));

//             // Write Preciseness
//             if (grabbableObject != null && unloadingArea != null)
//             {
//                 preciseness = Vector3.Distance(grabbableObject.position, unloadingArea.position) / 10; // Divided by 10 because of scaling differences between Unity and real-world
//             }
//             else
//             {
//                 Debug.LogError("GrabbableObject or UnloadingArea is not assigned!");
//             }

//             writer.WriteLine("Preciseness: " + preciseness.ToString("F2") + " meters");
//             writer.WriteLine(new string('-', 40));

//             // Write collision logs
//             writer.WriteLine("Collision Logs:");
//             foreach (string log in collisionLogs)
//             {
//                 writer.WriteLine(log);
//             }
//         }
//     }
// }







// // Metrics calculator before adding collision logs
// using System.Collections.Generic;
// using System.IO;
// using UnityEngine;

// public class Metrics_Calculator : MonoBehaviour
// {
//     public Transform grabbableObject; // Assign the grabbable object in the Inspector
//     public Transform unloadingArea; // Assign the unloading area in the Inspector
//     public Transform loadSwayMeasurementReferencePoint; // Assign the reference point in the Inspector

//     private Transform craneHook;
//     private Vector3 previousPosition;
//     private bool loadingStarted = false;
//     private bool firstFrameAfterLoadingStarted = true;
//     private float totalPathLength = 0f;
//     private List<Vector3> pathCoordinates = new List<Vector3>(); // Record positions for the path
//     private float elapsedTime = 0f;
//     private float measurementInterval = 1f; // Time interval in seconds
//     private string desktopPath = "C:/Users/mzk6120/Desktop";
//     private string metricsFilePath;
//     private string csvFilePath;

//     private float preciseness;

//     private List<float> loadSwayMeasurements = new List<float>(); // Record sway measurements

//     public MoveObject moveObject;

//     private float taskStartTime;
//     private float taskEndTime;
//     private float taskCompletionTime;

//     public CSVWriter csvWriter; // Assign this in the Inspector

//     void Start()
//     {
//         craneHook = transform;

//         string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");
//         metricsFilePath = Path.Combine(desktopPath, $"Metrics_{timestamp}.txt");
//         csvFilePath = Path.Combine(desktopPath, $"LoadData_{timestamp}.csv");

//         // Initialize CSV file with headers
//         csvWriter.WriteHeader("Time,X,Y,Z,Sway,LoadingStarted,SignalASlewingAngleRotaryEncoder,SignalBSlewingAngleRotaryEncoder,SlewingAngle,SignalARadiusLinearEncoder,SignalBRadiusLinearEncoder,Radius,SignalAHookHeightLinearEncoder,SignalBHookHeightLinearEncoder,HookHeight");

//         // Subscribe to the event
//         EncoderSignalChangeNotifier.OnSignalAChanged += WriteMetrics;
//     }

//     void OnDestroy()
//     {
//         // Unsubscribe from the event
//         EncoderSignalChangeNotifier.OnSignalAChanged -= WriteMetrics;
//     }

//     private void WriteMetrics()
//     {
//         float sway = 0f;
//         if (grabbableObject != null && grabbableObject.parent == craneHook)
//         {
//             sway = Vector3.Distance(grabbableObject.position, loadSwayMeasurementReferencePoint.position);
//             loadSwayMeasurements.Add(sway);
//         }

//         // Update the centralized CSV writer with the latest metrics
//         csvWriter.UpdateMetrics(Time.time, grabbableObject.position.x, grabbableObject.position.y, grabbableObject.position.z, sway, loadingStarted);
//     }

//     void FixedUpdate()
//     {
//         taskStartTime = crane_animate2.taskStartTime;
//         taskEndTime = DustCloudTrigger.taskEndTime;

//         loadingStarted = !moveObject.canGrab;
//         // Calculate path length during loading and save the result
//         if (loadingStarted)
//         {
//             if (firstFrameAfterLoadingStarted)
//             {
//                 previousPosition = craneHook.position;
//                 firstFrameAfterLoadingStarted = false;
//             }

//             elapsedTime += Time.deltaTime;

//             // Measure path length and record load position and sway at the specified interval
//             if (elapsedTime >= measurementInterval)
//             {
//                 float distance = Vector3.Distance(previousPosition, craneHook.position);
//                 totalPathLength += distance;
//                 pathCoordinates.Add(craneHook.position); // Record crane_hook position for the path
//                 previousPosition = craneHook.position;

//                 elapsedTime = 0f; // Reset elapsed time after recording data
//             }
//         }
//     }

//     public void ExportMetrics()
//     {
//         taskCompletionTime = taskEndTime - taskStartTime;

//         using (StreamWriter writer = new StreamWriter(metricsFilePath))
//         {
//             writer.WriteLine("Path Length: " + (totalPathLength/2).ToString("F2") + " meters"); // totalPathLength is divided by 2 because of scaling differences between Unity and real-world

//             // Add a separator line
//             writer.WriteLine(new string('-', 40));

//             // Write path coordinates to the file
//             writer.WriteLine("Path Coordinates:");
//             foreach (Vector3 coordinate in pathCoordinates)
//             {
//                 writer.WriteLine(coordinate.ToString());
//             }

//             writer.WriteLine(new string('-', 40));

//             // Write task completion time to the file
//             writer.WriteLine("Task Completion Time: " + taskCompletionTime.ToString("F2") + " seconds");

//             writer.WriteLine(new string('-', 40));

//             // Write Preciseness
//             if (grabbableObject != null && unloadingArea != null)
//             {             
//                 preciseness = Vector3.Distance(grabbableObject.position, unloadingArea.position) / 10; // Divided by 10 because of scaling differences between Unity and real-world
//             }
//             else
//             {
//                 Debug.LogError("GrabbableObject or UnloadingArea is not assigned!");
//             }

//             writer.WriteLine("Preciseness: " + preciseness.ToString("F2") + " meters");
//             writer.WriteLine(new string('-', 40));
//         }
//     }
// }




/// Metrics calculator before combining the csv outputs into one export
// using System.Collections;
// using System.Collections.Generic;
// using System.IO;
// using UnityEngine;

// public class Metrics_Calculator : MonoBehaviour
// {
//     public Transform grabbableObject; // Assign the grabbable object in the Inspector
//     public Transform unloadingArea; // Assign the unloading area in the Inspector
//     public Transform loadSwayMeasurementReferencePoint; // Assign the reference point in the Inspector

//     private Transform craneHook;
//     private Vector3 previousPosition;
//     private bool loadingStarted = false;
//     private bool firstFrameAfterLoadingStarted = true;
//     private float totalPathLength = 0f;
//     private List<Vector3> pathCoordinates = new List<Vector3>(); // Record positions for the path
//     private float elapsedTime = 0f;
//     private float measurementInterval = 1f; // Time interval in seconds
//     private string desktopPath = "C:/Users/mzk6120/Desktop";
//     private string metricsFilePath;
//     private string csvFilePath;

//     private float preciseness;

//     private List<float> loadSwayMeasurements = new List<float>(); // Record sway measurements

//     public MoveObject moveObject;

//     private float taskStartTime;
//     private float taskEndTime;
//     private float taskCompletionTime;

//     void Start()
//     {
//         craneHook = transform;

//         string timestamp = System.DateTime.Now.ToString("yyyyMMddHHmmss");
//         metricsFilePath = Path.Combine(desktopPath, $"Metrics_{timestamp}.txt");
//         csvFilePath = Path.Combine(desktopPath, $"LoadData_{timestamp}.csv");

//         // Initialize CSV file with headers
//         using (StreamWriter writer = new StreamWriter(csvFilePath))
//         {
//             writer.WriteLine("Time,X,Y,Z,Sway");
//         }
//     }

//     void FixedUpdate()
//     {
//         taskStartTime = crane_animate2.taskStartTime;
//         taskEndTime = DustCloudTrigger.taskEndTime;

//         loadingStarted = !moveObject.canGrab;
//         // Calculate path length during loading and save the result
//         if (loadingStarted)
//         {
//             if (firstFrameAfterLoadingStarted)
//             {
//                 previousPosition = craneHook.position;
//                 firstFrameAfterLoadingStarted = false;
//             }

//             elapsedTime += Time.deltaTime;

//             // Measure path length and record load position and sway at the specified interval
//             if (elapsedTime >= measurementInterval)
//             {
//                 float distance = Vector3.Distance(previousPosition, craneHook.position);
//                 totalPathLength += distance;
//                 pathCoordinates.Add(craneHook.position); // Record crane_hook position for the path
//                 previousPosition = craneHook.position;

//                 float sway = 0f;
//                 if (grabbableObject != null && grabbableObject.parent == craneHook)
//                 {
//                     sway = Vector3.Distance(grabbableObject.position, loadSwayMeasurementReferencePoint.position);
//                     loadSwayMeasurements.Add(sway);
//                 }

//                 using (StreamWriter writer = new StreamWriter(csvFilePath, true))
//                 {
//                     writer.WriteLine($"{Time.time:F2},{grabbableObject.position.x:F2},{grabbableObject.position.y:F2},{grabbableObject.position.z:F2},{sway:F2}");
//                 }

//                 elapsedTime = 0f; // Reset elapsed time after recording data
//             }
//         }
//     }

//     public void ExportMetrics()
//     {
//         taskCompletionTime = taskEndTime - taskStartTime;

//         using (StreamWriter writer = new StreamWriter(metricsFilePath))
//         {
//             writer.WriteLine("Path Length: " + totalPathLength.ToString("F2") + " meters");

//             // Add a separator line
//             writer.WriteLine(new string('-', 40));

//             // Write path coordinates to the file
//             writer.WriteLine("Path Coordinates:");
//             foreach (Vector3 coordinate in pathCoordinates)
//             {
//                 writer.WriteLine(coordinate.ToString());
//             }

//             writer.WriteLine(new string('-', 40));

//             // Write task completion time to the file
//             writer.WriteLine("Task Completion Time: " + taskCompletionTime.ToString("F2") + " seconds");

//             writer.WriteLine(new string('-', 40));

//             // Write Preciseness
//             if (grabbableObject != null && unloadingArea != null)
//             {
//                 preciseness = Vector3.Distance(grabbableObject.position, unloadingArea.position);
//             }
//             else
//             {
//                 Debug.LogError("GrabbableObject or UnloadingArea is not assigned!");
//             }

//             writer.WriteLine("Preciseness: " + preciseness.ToString("F2") + " meters");
//             writer.WriteLine(new string('-', 40));
//         }
//     }
// }





/// Performance metrics with uncompleted "clearance" metric

// using System.Collections;
// using System.Collections.Generic;
// using System.IO;
// using UnityEngine;

// public class Metrics_Calculator : MonoBehaviour
// {
//     public Transform grabbableObject; // Assign the grabbable object in the Inspector
//     public Transform unloadingArea; // Assign the unloading area in the Inspector
//     public Transform loadSwayMeasurementReferencePoint; // Assign the reference point in the Inspector
//     public GameObject building; // Assign the building object in the Inspector

//     private Transform craneHook;
//     private Vector3 previousPosition;
//     private bool loadingStarted = false;
//     private bool firstFrameAfterLoadingStarted = true;
//     private float totalPathLength = 0f;
//     private List<Vector3> pathPositions = new List<Vector3>(); // Record positions for the path
//     private bool exportedMetrics = false;
//     private float elapsedTime = 0f;
//     private float measurementInterval = 1f; // Time interval in seconds
//     private string desktopPath = "C:/Users/mzk6120/Desktop";
//     private string filePath;

//     private float preciseness;
//     private float precisenessStableTime = 20f; // You can adjust this time as needed
//     private float elapsedTimeAfterUnloading = 0f;

//     private float loadSway = 0f;
//     private List<float> loadSwayMeasurements = new List<float>(); // Record sway measurements

//     private float closestDistance = float.MaxValue;
//     private List<float> clearanceMeasurements = new List<float>(); // Record clearance measurements

//     public MoveObject moveObject;

//     void Start()
//     {
//         craneHook = transform;
//     }

//     void FixedUpdate()
//     {
//         loadingStarted = !moveObject.canGrab;

//         // Calculate path length during loading and save the result
//         if (loadingStarted)
//         {
//             if (firstFrameAfterLoadingStarted)
//             {
//                 previousPosition = craneHook.position;
//                 firstFrameAfterLoadingStarted = false;
//             }

//             elapsedTime += Time.deltaTime;

//             if (elapsedTime >= measurementInterval)
//             {
//                 float distance = Vector3.Distance(previousPosition, craneHook.position);
//                 totalPathLength += distance;
//                 pathPositions.Add(craneHook.position); // Record crane_hook position for the path
//                 previousPosition = craneHook.position;
//                 elapsedTime = 0f;
//             }

//             // Calculate load sway
//             if (grabbableObject != null && grabbableObject.parent == craneHook)
//             {
//                 loadSway = Vector3.Distance(grabbableObject.position, loadSwayMeasurementReferencePoint.position);
//                 loadSwayMeasurements.Add(loadSway);
//             }

//             // Calculate the closest distance to the building
//             Collider buildingCollider = building.GetComponent<Collider>();
//             Collider grabbableObjectCollider = grabbableObject.GetComponent<Collider>();
//             Vector3 closestPointOnBuilding = buildingCollider.ClosestPoint(grabbableObjectCollider.transform.position);
//             Vector3 closestPointOnGrabbableObject = grabbableObjectCollider.ClosestPoint(closestPointOnBuilding);
//             float distanceToBuilding = Vector3.Distance(closestPointOnGrabbableObject, closestPointOnBuilding);

//             if (distanceToBuilding < closestDistance)
//             {
//                 closestDistance = distanceToBuilding;
//             }
//             clearanceMeasurements.Add(closestDistance);
//         }

//         if (!loadingStarted && moveObject.unloadingFinished && !exportedMetrics)
//         {
//             // Export metrics to a .txt file
//             ExportMetricsToFile();

//             // Set the flag to indicate that metrics have been exported
//             exportedMetrics = true;
//         }
//     }

//     private void ExportMetricsToFile()
//     {
//         filePath = Path.Combine(desktopPath, "Metrics.txt");
//         using (StreamWriter writer = new StreamWriter(filePath))
//         {
//             writer.WriteLine("Path Length: " + totalPathLength.ToString("F2") + " meters");

//             // Add a separator line
//             writer.WriteLine(new string('-', 40));

//             // Write path positions to the file
//             writer.WriteLine("Path Positions:");
//             foreach (Vector3 position in pathPositions)
//             {
//                 writer.WriteLine(position.ToString());
//             }

//             writer.WriteLine(new string('-', 40));

//             // Write load sway measurements to the file
//             writer.WriteLine("Load Sway Measurements:");
//             foreach (float sway in loadSwayMeasurements)
//             {
//                 writer.WriteLine(sway.ToString("F2"));
//             }

//             writer.WriteLine(new string('-', 40));

//             // Write clearance measurements to the file
//             writer.WriteLine("Clearance Measurements:");
//             foreach (float clearance in clearanceMeasurements)
//             {
//                 writer.WriteLine(clearance.ToString("F2"));
//             }

//             writer.WriteLine(new string('-', 40));

//             // Start the coroutine and provide a callback to be executed when it finishes
//             StartCoroutine(MeasurePreciseness(() =>
//             {
//                 using (StreamWriter writerWithCallback = new StreamWriter(filePath, true))
//                 {
//                     // Write Preciseness
//                     writerWithCallback.WriteLine("Preciseness: " + preciseness.ToString("F2") + " meters");

//                     writerWithCallback.WriteLine(new string('-', 40));
//                 }
//             }));
//         }
//     }

//     IEnumerator MeasurePreciseness(System.Action onFinishCallback)
//     {
//         while (elapsedTimeAfterUnloading < precisenessStableTime)
//         {
//             // Wait for the object to stabilize on the unloading area
//             elapsedTimeAfterUnloading += Time.deltaTime;
//             yield return null;
//         }

//         // Measure the distance between the center point of the grabbableObject and the unloadingArea
//         if (grabbableObject != null && unloadingArea != null)
//         {
//             preciseness = Vector3.Distance(grabbableObject.position, unloadingArea.position);
//         }
//         else
//         {
//             Debug.LogError("GrabbableObject or UnloadingArea is not assigned!");
//         }

//         // Execute the callback to notify that the coroutine has finished
//         onFinishCallback?.Invoke();
//     }
// }