import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class RecommendationSystem {
    public static void main(String[] args) {
        try {
            // Load dataset (user-item interactions)
            DataModel model = new FileDataModel(new File("data/ratings.csv"));

            // Compute user similarity using Pearson correlation
            UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

            // Define user neighborhood
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

            // Build recommender system
            Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

            // Generate recommendations for a specific user (e.g., user ID 1)
            List<RecommendedItem> recommendations = recommender.recommend(1, 3);

            // Print recommendations
            for (RecommendedItem recommendation : recommendations) {
                System.out.println("Recommended Item ID: " + recommendation.getItemID() +
                        " | Estimated Rating: " + recommendation.getValue());
            }

            // Evaluate the recommender system
            RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
            double score = evaluator.evaluate(recommender, null, model, 0.9, 1.0);
            System.out.println("Evaluation Score: " + score);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
