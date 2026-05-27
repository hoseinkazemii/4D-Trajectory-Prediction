using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class PositionAssignment : MonoBehaviour
{
    // List of predefined (X, Y, Z) positions for grabbableObject and unloadingArea
    public List<Vector3> grabbableObjectPositions = new List<Vector3>();
    public List<Vector3> unloadingAreaPositions = new List<Vector3>();

    // Enum for dropdown selection
    public enum GrabbableObjectIndex { Position0, PositionForPLScenario, Position1, Position2, Position3, Position4, Position5 }
    public enum UnloadingAreaIndex { Position0, PositionForPLScenario, Position1, Position2, Position3, Position4, Position5 }

    // Drop-down menu variables
    public GrabbableObjectIndex grabbableObjectIndex = GrabbableObjectIndex.Position0;
    public UnloadingAreaIndex unloadingAreaIndex = UnloadingAreaIndex.Position0;

    // Variables to assign in the Inspector for the grabbableObject and unloadingArea
    public GameObject grabbableObject;
    public GameObject unloadingArea;
    public GameObject unloadingStorey;

    // Store default positions
    private Vector3 defaultGrabbableObjectPosition;
    private Vector3 defaultUnloadingAreaPosition;

    // Layer names
    private string defaultUnloadingStoreyLayer = "UnloadingStorey";
    private string unloadingStorey4DTrajectoryLayer = "UnloadingStorey4DTrajectory";

    // This method is called when you change the index in the Inspector or modify the script in the editor
    void OnValidate()
    {
        InitializePositions();  // Ensure the positions are initialized even in edit mode
        UpdatePositions();  // Update the object positions based on the selected index
    }

    // Initialize the predefined positions for the drop-down menu
    void InitializePositions()
    {
        // Clear the lists to prevent duplicate entries
        grabbableObjectPositions.Clear();
        unloadingAreaPositions.Clear();

        // Default positions
        defaultGrabbableObjectPosition = new Vector3(-42.5999985f, 0.299804211f, 52.0999985f);
        defaultUnloadingAreaPosition = new Vector3(-78.5999985f, 47f, -37f);

        // Add positions for grabbableObject
        grabbableObjectPositions.Add(defaultGrabbableObjectPosition);
        grabbableObjectPositions.Add(new Vector3(-35.7781067f,0.178566933f,59.5939827f));
        grabbableObjectPositions.Add(new Vector3(-45.8101273f, -2.55567002f, 22.5101185f));
        grabbableObjectPositions.Add(new Vector3(-51.25f, 4.37500143f, 43.1199989f));
        grabbableObjectPositions.Add(new Vector3(-50.3612366f, 4.37500143f, 32.5635605f));
        grabbableObjectPositions.Add(new Vector3(-20.9088898f,0.290589809f,37.57938f));
        grabbableObjectPositions.Add(new Vector3(-53.1199989f, 4.37499952f, 38.5299988f));

        // Add positions for unloadingArea
        unloadingAreaPositions.Add(defaultUnloadingAreaPosition);
        unloadingAreaPositions.Add(new Vector3(-78.5999985f, 47f, -37f));
        unloadingAreaPositions.Add(new Vector3(-78.5999985f, 47.0999985f, 30.6200008f));
        unloadingAreaPositions.Add(new Vector3(27.8999996f, 26.7999992f, 80.5999985f));
        unloadingAreaPositions.Add(new Vector3(64.0999985f, 26.7999992f, -33.2999992f));
        unloadingAreaPositions.Add(new Vector3(37.4199982f, 26.6000004f, -115.879997f));
        unloadingAreaPositions.Add(new Vector3(-63f, 12.1000004f, -82.5f));
    }

    // Update positions based on the chosen drop-down selection
    void UpdatePositions()
    {
        // Convert the enum to int (underlying index)
        int grabIndex = (int)grabbableObjectIndex;
        int unloadIndex = (int)unloadingAreaIndex;

        // Set the positions based on the chosen indices from drop-downs
        grabbableObject.transform.localPosition = grabbableObjectPositions[grabIndex];
        unloadingArea.transform.localPosition = unloadingAreaPositions[unloadIndex];

        // Set layer based on the position selection
        if (grabbableObjectIndex == GrabbableObjectIndex.Position0 || grabbableObjectIndex == GrabbableObjectIndex.PositionForPLScenario
        || unloadingAreaIndex == UnloadingAreaIndex.Position0 || unloadingAreaIndex == UnloadingAreaIndex.PositionForPLScenario)
        {
            unloadingStorey.layer = LayerMask.NameToLayer(defaultUnloadingStoreyLayer);
        }
        else
        {
            unloadingStorey.layer = LayerMask.NameToLayer(unloadingStorey4DTrajectoryLayer);
        }
    }
}



// using System.Collections.Generic;  // For using List
// using UnityEngine;

// [ExecuteInEditMode]  // This allows the script to run in the Unity editor while editing
// public class PositionAssignment : MonoBehaviour
// {
//     // List of predefined (X, Y, Z) positions for grabbableObject and unloadingArea
//     public List<Vector3> grabbableObjectPositions = new List<Vector3>();
//     public List<Vector3> unloadingAreaPositions = new List<Vector3>();

//     // Enum for dropdown selection
//     public enum GrabbableObjectIndex { Position0, PositionForPLScenario, Position1, Position2, Position3, Position4, Position5 }
//     public enum UnloadingAreaIndex { Position0, PositionForPLScenario, Position1, Position2, Position3, Position4, Position5 }

//     // Drop-down menu variables
//     public GrabbableObjectIndex grabbableObjectIndex = GrabbableObjectIndex.Position0;
//     public UnloadingAreaIndex unloadingAreaIndex = UnloadingAreaIndex.Position0;

//     // Variables to assign in the Inspector for the grabbableObject and unloadingArea
//     public GameObject grabbableObject;
//     public GameObject unloadingArea;

//     // Store default positions
//     private Vector3 defaultGrabbableObjectPosition;
//     private Vector3 defaultUnloadingAreaPosition;

//     // This method is called when you change the index in the Inspector or modify the script in the editor
//     void OnValidate()
//     {
//         InitializePositions();  // Ensure the positions are initialized even in edit mode
//         UpdatePositions();  // Update the object positions based on the selected index
//     }

//     // Initialize the predefined positions for the drop-down menu
//     void InitializePositions()
//     {
//         // Clear the lists to prevent duplicate entries
//         grabbableObjectPositions.Clear();
//         unloadingAreaPositions.Clear();

//         // Default positions
//         defaultGrabbableObjectPosition = new Vector3(-42.5999985f, 0.299804211f, 52.0999985f);
//         defaultUnloadingAreaPosition = new Vector3(-78.5999985f, 47f, -37f);

//         // Add positions for grabbableObject
//         grabbableObjectPositions.Add(defaultGrabbableObjectPosition);
//         grabbableObjectPositions.Add(new Vector3(-35.7781067f,0.178566933f,59.5939827f));
//         grabbableObjectPositions.Add(new Vector3(-53.1199989f, 4.37499952f, 38.5299988f));
//         grabbableObjectPositions.Add(new Vector3(-45.8101273f, -2.55567002f, 22.5101185f));
//         grabbableObjectPositions.Add(new Vector3(-51.25f, 4.37500143f, 43.1199989f));
//         grabbableObjectPositions.Add(new Vector3(-50.3612366f, 4.37500143f, 32.5635605f));
//         grabbableObjectPositions.Add(new Vector3(-24.0962429f, 0.381225586f, 37.5809898f));

//         // Add positions for unloadingArea
//         unloadingAreaPositions.Add(defaultUnloadingAreaPosition);
//         unloadingAreaPositions.Add(new Vector3(-78.5999985f, 47f, -37f));
//         unloadingAreaPositions.Add(new Vector3(-63f, 12.1000004f, -82.5f));
//         unloadingAreaPositions.Add(new Vector3(-78.5999985f, 47.0999985f, 30.6200008f));
//         unloadingAreaPositions.Add(new Vector3(27.8999996f, 26.7999992f, 80.5999985f));
//         unloadingAreaPositions.Add(new Vector3(64.0999985f, 26.7999992f, -33.2999992f));
//         unloadingAreaPositions.Add(new Vector3(37.4199982f, 26.6000004f, -115.879997f));
//     }

//     // Update positions based on the chosen drop-down selection
//     void UpdatePositions()
//     {
//         // Convert the enum to int (underlying index)
//         int grabIndex = (int)grabbableObjectIndex;
//         int unloadIndex = (int)unloadingAreaIndex;

//         // Set the positions based on the chosen indices from drop-downs
//         grabbableObject.transform.localPosition = grabbableObjectPositions[grabIndex];
//         unloadingArea.transform.localPosition = unloadingAreaPositions[unloadIndex];
//     }
// }